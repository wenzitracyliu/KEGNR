import argparse

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import OpenMatch as om
import src
import jsonlines


def dev(args, model, metric, dev_loader, device):
    rst_dict = {}
    for dev_batch in dev_loader:
        query_id, doc_id, label = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            elif args.model == 'graph':
                batch_score, _, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device),
                                          dev_batch['segment_ids'].to(device), dev_batch['token_starts'].to(device),
                                          dev_batch['q_term_nodes'].to(device), dev_batch['doc_term_nodes'].to(device),
                                          dev_batch['section'].to(device), dev_batch['sen_len'].to(device),
                                          dev_batch['doc_sen_idxes'].to(device), dev_batch['rgcn_adjacency'].to(device))
            elif args.model == 'roberta':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device))
            elif args.model == 'edrm':
                batch_score, _ = model(dev_batch['query_wrd_idx'].to(device), dev_batch['query_wrd_mask'].to(device),
                                       dev_batch['doc_wrd_idx'].to(device), dev_batch['doc_wrd_mask'].to(device),
                                       dev_batch['query_ent_idx'].to(device), dev_batch['query_ent_mask'].to(device),
                                       dev_batch['doc_ent_idx'].to(device), dev_batch['doc_ent_mask'].to(device),
                                       dev_batch['query_des_idx'].to(device), dev_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(dev_batch['query_idx'].to(device), dev_batch['query_mask'].to(device),
                                       dev_batch['doc_idx'].to(device), dev_batch['doc_mask'].to(device))
            if args.task == 'classification':
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)

            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    return rst_dict


def train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader, dev_loader, device):
    best_mes = 0.0
    with torch.no_grad():
        rst_dict = dev(args, model, metric, dev_loader, device)
        om.utils.save_trec(args.res, rst_dict)
        if args.metric.split('_')[0] == 'mrr':
            mes = metric.get_mrr(args.qrels, args.res, args.metric)
        else:
            mes = metric.get_metric(args.qrels, args.res, args.metric)
    if mes >= best_mes:
        best_mes = mes
        print('save_model...')
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), args.save)
        else:
            torch.save(model.state_dict(), args.save)
    print('initial result: ', mes)
    last_mes = mes
    for epoch in range(args.epoch):
        avg_loss = 0.0
        log_prob_ps = []
        log_prob_ns = []
        for step, train_batch in enumerate(train_loader):
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_pos_wrd_idx'].to(device), train_batch['doc_pos_wrd_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            batch_probs = F.gumbel_softmax(batch_probs, tau=args.tau)
            m = Categorical(batch_probs)
            action = m.sample()
            if action.sum().item() < 1:
                # m_scheduler.step()
                if (step+1) % args.eval_every == 0 and len(log_prob_ps) > 0:
                    with torch.no_grad():
                        rst_dict = dev(args, model, metric, dev_loader, device)
                        print(rst_dict)
                        om.utils.save_trec(args.res, rst_dict)
                        if args.metric.split('_')[0] == 'mrr':
                            mes = metric.get_mrr(args.qrels, args.res, args.metric)
                        else:
                            mes = metric.get_metric(args.qrels, args.res, args.metric)
                    if mes >= best_mes:
                        best_mes = mes
                        print('save_model...')
                        if torch.cuda.device_count() > 1:
                            torch.save(model.module.state_dict(), args.save)
                        else:
                            torch.save(model.state_dict(), args.save)
                    print(step+1, avg_loss/len(log_prob_ps), mes, best_mes)
                    avg_loss = 0.0

                    reward = mes - last_mes
                    last_mes = mes
                    if reward >= 0:
                        policy_loss = [(-log_prob_p * reward).sum().unsqueeze(-1) for log_prob_p in log_prob_ps]
                    else:
                        policy_loss = [(log_prob_n * reward).sum().unsqueeze(-1) for log_prob_n in log_prob_ns]
                    policy_loss = torch.cat(policy_loss).sum()
                    policy_loss.backward()
                    p_optim.step()
                    p_optim.zero_grad()

                    if args.reset:
                        state_dict = torch.load(args.save)
                        model.load_state_dict(state_dict)
                        last_mes = best_mes
                    log_prob_ps = []
                    log_prob_ns = []
                continue

            filt = action.nonzero().squeeze(-1).cpu()
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].index_select(0, filt).to(device),
                                               train_batch['input_mask_pos'].index_select(0, filt).to(device),
                                               train_batch['segment_ids_pos'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].index_select(0, filt).to(device),
                                               train_batch['input_mask_neg'].index_select(0, filt).to(device),
                                               train_batch['segment_ids_neg'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].index_select(0, filt).to(device),
                                           train_batch['input_mask'].index_select(0, filt).to(device),
                                           train_batch['segment_ids'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')

            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].index_select(0, filt).to(device), train_batch['input_mask_pos'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].index_select(0, filt).to(device), train_batch['input_mask_neg'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].index_select(0, filt).to(device), train_batch['input_mask'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_pos_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_ent_idx'].index_select(0, filt).to(device), train_batch['doc_pos_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_pos_des_idx'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_neg_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_ent_idx'].index_select(0, filt).to(device), train_batch['doc_neg_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_neg_des_idx'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_wrd_mask'].index_select(0, filt).to(device),
                                           train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_ent_idx'].index_select(0, filt).to(device), train_batch['doc_ent_mask'].index_select(0, filt).to(device),
                                           train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_des_idx'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_idx'].index_select(0, filt).to(device), train_batch['doc_pos_mask'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_idx'].index_select(0, filt).to(device), train_batch['doc_neg_mask'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_idx'].index_select(0, filt).to(device), train_batch['doc_mask'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')

            mask = action.ge(0.5)
            log_prob_p = m.log_prob(action)
            log_prob_n = m.log_prob(1-action)
            log_prob_ps.append(torch.masked_select(log_prob_p, mask))
            log_prob_ns.append(torch.masked_select(log_prob_n, mask))

            if args.task == 'ranking':
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            if torch.cuda.device_count() > 1:
                batch_loss = batch_loss.mean(-1)
            batch_loss = batch_loss.mean()
            avg_loss += batch_loss.item()
            batch_loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()

            if (step+1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = dev(args, model, metric, dev_loader, device)
                    om.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)
                if mes >= best_mes:
                    best_mes = mes
                    print('save_model...')
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.save)
                    else:
                        torch.save(model.state_dict(), args.save)

                print(step+1, avg_loss/len(log_prob_ps), mes, best_mes)
                avg_loss = 0.0

                reward = mes - last_mes
                last_mes = mes
                if reward >= 0:
                    policy_loss = [(-log_prob_p * reward).sum().unsqueeze(-1) for log_prob_p in log_prob_ps]
                else:
                    policy_loss = [(log_prob_n * reward).sum().unsqueeze(-1) for log_prob_n in log_prob_ns]
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                p_optim.step()
                p_optim.zero_grad()

                if args.reset:
                    state_dict = torch.load(args.save)
                    model.load_state_dict(state_dict)
                    last_mes = best_mes
                log_prob_ps = []
                log_prob_ns = []


def train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device):
    best_mes = 0.0
    for epoch in range(args.epoch):
        avg_loss = 0.0

        for step, train_batch in enumerate(train_loader):
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'graph':
                if args.task == 'ranking':
                    batch_score_pos, _, _ = model(train_batch['input_ids_pos'].to(device),
                                                  train_batch['input_mask_pos'].to(device),
                                                  train_batch['segment_ids_pos'].to(device),
                                                  train_batch['token_starts_pos'].to(device),
                                                  train_batch['q_term_nodes_pos'].to(device),
                                                  train_batch['doc_term_nodes_pos'].to(device),
                                                  train_batch['section_pos'].to(device),
                                                  train_batch['sen_len_pos'].to(device),
                                                  train_batch['doc_sen_idxes_pos'].to(device),
                                                  train_batch['rgcn_adjacency_pos'].to(device))

                    batch_score_neg, _, _ = model(train_batch['input_ids_neg'].to(device),
                                                  train_batch['input_mask_neg'].to(device),
                                                  train_batch['segment_ids_neg'].to(device),
                                                  train_batch['token_starts_neg'].to(device),
                                                  train_batch['q_term_nodes_neg'].to(device),
                                                  train_batch['doc_term_nodes_neg'].to(device),
                                                  train_batch['section_neg'].to(device),
                                                  train_batch['sen_len_neg'].to(device),
                                                  train_batch['doc_sen_idxes_neg'].to(device),
                                                  train_batch['rgcn_adjacency_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device),
                                              train_batch['segment_ids'].to(device), train_batch['token_starts'].to(device),
                                              train_batch['q_term_nodes'].to(device), train_batch['doc_term_nodes'].to(device),
                                              train_batch['section'].to(device), train_batch['sen_len'].to(device),
                                              train_batch['doc_sen_idxes'].to(device), train_batch['rgcn_adjacency'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                               train_batch['doc_pos_wrd_idx'].to(device), train_batch['doc_pos_wrd_mask'].to(device),
                                               train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                               train_batch['doc_pos_ent_idx'].to(device), train_batch['doc_pos_ent_mask'].to(device),
                                               train_batch['query_des_idx'].to(device), train_batch['doc_pos_des_idx'].to(device))
                    batch_score_neg, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                               train_batch['doc_neg_wrd_idx'].to(device), train_batch['doc_neg_wrd_mask'].to(device),
                                               train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                               train_batch['doc_neg_ent_idx'].to(device), train_batch['doc_neg_ent_mask'].to(device),
                                               train_batch['query_des_idx'].to(device), train_batch['doc_neg_des_idx'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                           train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device),
                                           train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                           train_batch['doc_ent_idx'].to(device), train_batch['doc_ent_mask'].to(device),
                                           train_batch['query_des_idx'].to(device), train_batch['doc_des_idx'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                    batch_score_neg, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_neg_idx'].to(device), train_batch['doc_neg_mask'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            # print(batch_score)
            if args.task == 'ranking':
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            if torch.cuda.device_count() > 1:
                batch_loss = batch_loss.mean()
            # print(batch_loss)
            avg_loss += batch_loss.item()
            batch_loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()

            if (step+1) % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    rst_dict = dev(args, model, metric, dev_loader, device)
                    om.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)

                model.train()
                if mes >= best_mes:
                    best_mes = mes
                    print('save_model...')
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.save)
                    else:
                        torch.save(model.state_dict(), args.save)
                        # for name, weight in model.state_dict().items():
                        #     print(name)
                print(step+1, avg_loss/args.eval_every, mes, best_mes)
                avg_loss = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-reset', action='store_true', default=False)
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-eval_every', type=int, default=1000)
    parser.add_argument('-label_num', type=int, default=2)
    parser.add_argument('-seed', type=int, default=2021)
    parser.add_argument('-encoder_type', type=str, default='bert')
    parser.add_argument('-seq_max_len', type=int, default=256)
    parser.add_argument('-graph_hidden_dim', type=int, default=256)
    parser.add_argument('-graph_layer_nums', type=int, default=3)
    parser.add_argument('-relation_cnt', type=int, default=5)
    parser.add_argument('-keep_ratio', type=float, default=0.5)

    args = parser.parse_args()

    om.set_environment(args.seed, True)
    print('training ', args.model)
    print(args.save)
    args.model = args.model.lower()
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        if args.maxp:
            train_set = om.data.datasets.BertMaxPDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            train_set = om.data.datasets.BertDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        print('reading dev data...')
        if args.maxp:
            dev_set = om.data.datasets.BertMaxPDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            dev_set = om.data.datasets.BertDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
    elif args.model == 'graph':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        train_set = src.GraphDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            encoder_type=args.encoder_type,
            seq_max_len=args.seq_max_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = src.GraphDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            encoder_type=args.encoder_type,
            seq_max_len=args.seq_max_len,
            max_input=args.max_input,
            task=args.task
        )

    elif args.model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        train_set = om.data.datasets.RobertaDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.RobertaDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
    elif args.model == 'edrm':
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        ent_tokenizer = om.data.tokenizers.WordTokenizer(
            vocab=args.ent_vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.EDRMDataset(
            dataset=args.train,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.EDRMDataset(
            dataset=args.dev,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
    else:
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.Dataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.Dataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
    train_loader = om.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )
    dev_loader = om.data.DataLoader(
        dataset=dev_set,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=8
    )

    if args.model == 'bert' or args.model == 'roberta':
        if args.maxp:
            model = om.models.BertMaxP(
                pretrained=args.pretrain,
                max_query_len=args.max_query_len,
                max_doc_len=args.max_doc_len,
                mode=args.mode,
                task=args.task
            )
        else:
            model = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task=args.task
            )
        if args.reinfoselect:
            policy = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task='classification'
            )
    elif args.model == 'graph':
        model = src.GIR(
            pretrained=args.pretrain,
            graph_hidden_dim=args.graph_hidden_dim,
            graph_layer_nums=args.graph_layer_nums,
            relation_cnt=args.relation_cnt,
            keep_ratio=args.keep_ratio,
            task=args.task
        )
    elif args.model == 'edrm':
        model = om.models.EDRM(
            wrd_vocab_size=tokenizer.get_vocab_size(),
            ent_vocab_size=ent_tokenizer.get_vocab_size(),
            wrd_embed_dim=tokenizer.get_embed_dim(),
            ent_embed_dim=128,
            max_des_len=20,
            max_ent_num=3,
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            wrd_embed_matrix=tokenizer.get_embed_matrix(),
            ent_embed_matrix=None,
            task=args.task
        )
    elif args.model == 'tk':
        model = om.models.TK(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            head_num=10,
            hidden_dim=100,
            layer_num=2,
            kernel_num=args.n_kernels,
            dropout=0.0,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'cknrm':
        model = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'knrm':
        model = om.models.KNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    else:
        raise ValueError('model name error.')

    if args.reinfoselect and args.model != 'bert':
        policy = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task='classification'
        )

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        if args.model == 'bert':
            st = {}
            for k in state_dict:
                if k.startswith('bert'):
                    st['_model'+k[len('bert'):]] = state_dict[k]
                elif k.startswith('classifier'):
                    st['_dense'+k[len('classifier'):]] = state_dict[k]
                else:
                    st[k] = state_dict[k]
            model.load_state_dict(st)
        else:
            model.load_state_dict(state_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.reinfoselect:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    else:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1)
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size)
    if args.reinfoselect:
        p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)
    metric = om.metrics.Metric()

    model.to(device)
    model.train()
    if args.reinfoselect:
        policy.to(device)
    loss_fn.to(device)
    if torch.cuda.device_count() > 1:

        model = nn.DataParallel(model)
        loss_fn = nn.DataParallel(loss_fn)

    if args.reinfoselect:
        train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader, dev_loader, device)
    else:
        train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device)
        

if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-task', type=str, default='ranking')
    # parser.add_argument('-model', type=str, default='bert')
    # parser.add_argument('-reinfoselect', action='store_true', default=False)
    # parser.add_argument('-reset', action='store_true', default=False)
    # parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_toy.jsonl')
    # parser.add_argument('-max_input', type=int, default=1280000)
    # parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    # parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    # parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    # parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    # parser.add_argument('-ent_vocab', type=str, default='')
    # parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    # parser.add_argument('-checkpoint', type=str, default=None)
    # parser.add_argument('-res', type=str, default='./results/bert.trec')
    # parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    # parser.add_argument('-mode', type=str, default='cls')
    # parser.add_argument('-n_kernels', type=int, default=21)
    # parser.add_argument('-max_query_len', type=int, default=20)
    # parser.add_argument('-max_doc_len', type=int, default=150)
    # parser.add_argument('-maxp', action='store_true', default=False)
    # parser.add_argument('-epoch', type=int, default=1)
    # parser.add_argument('-batch_size', type=int, default=8)
    # parser.add_argument('-lr', type=float, default=2e-5)
    # parser.add_argument('-tau', type=float, default=1)
    # parser.add_argument('-n_warmup_steps', type=int, default=1000)
    # parser.add_argument('-eval_every', type=int, default=1000)
    # parser.add_argument('-label_num', type=int, default=2)
    # parser.add_argument('-seed', type=int, default=2021)
    # parser.add_argument('-encoder_type', type=str, default='bert')
    # parser.add_argument('-seq_max_len', type=int, default=256)
    # parser.add_argument('-graph_hidden_dim', type=int, default=256)
    # parser.add_argument('-graph_layer_nums', type=int, default=3)
    # parser.add_argument('-relation_cnt', type=int, default=5)
    # parser.add_argument('-keep_ratio', type=float, default=0.5)
    #
    # args = parser.parse_args()
    # tokenizer = AutoTokenizer.from_pretrained(args.vocab)
    # print('reading training data...')
    # train_set = src.GraphDataset(
    #     dataset=args.train,
    #     tokenizer=tokenizer,
    #     mode='train',
    #     encoder_type=args.encoder_type,
    #     seq_max_len=args.seq_max_len,
    #     max_input=args.max_input,
    #     task=args.task
    # )
    # train_loader = om.data.DataLoader(
    #     dataset=train_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=8
    # )
    # for step, train_batch in enumerate(train_loader):
    #     print(train_batch['rgcn_adjacency'][0][5].shape)
    #     break
