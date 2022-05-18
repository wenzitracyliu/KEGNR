import argparse
import json
import spacy
import pandas as pd
import numpy as np
import jsonlines
import re
import string


def locate_term(term, doc, sen_len):
    term_len = len(term)
    doc_len = len(doc)
    starts = []
    ends = []
    s_loc = []
    i = 0
    while i + term_len <= doc_len:
        if term == doc[i:i+term_len]:
            starts.append(i)
            ends.append(i+term_len)

        i += 1

    i = 0
    for start, end in zip(starts, ends):
        while sen_len[i] <= start:
            i += 1
        if end <= sen_len[i]:
            s_loc.append(i)
        else:
            print('wrong: ', term, doc, sen_len, i, start, end)

    return starts, ends, s_loc


def preprocessing(text, topic_kb, nlp):

    doc = nlp(text)
    query = []
    doc_list = list(map(lambda x: str(x), doc))
    doc_lemma_list = list(map(lambda x: x.lemma_.lower(), doc))
    sen = ' '.join(doc_list).split(' . ')

    s_len = [len(s.split(' ')) + 1 for s in sen]
    s_len[-1] -= 1
    sen_len = [sum(s_len[:i+1]) for i, l in enumerate(s_len)]

    q_term_nodes = []
    doc_term_nodes = []
    sen_nodes = []
    q_doc_dict = {}  # use to record edges of query term nodes and document term nodes
    doc_term_dict = {}
    query_id_dict = {}
    # Query Term Node [id, start, end, 1000, node_type_id]
    # Disease Term [id, start, end, sen_loc, node_type_id]
    i = 0
    q = set()
    query_len = 0
    disease = topic_kb['disease']
    query_id_dict[disease] = i
    disease_list = disease.split(' ')
    query.extend(disease_list)
    q.update([disease.lower()])
    q_term_nodes.append([i, query_len, query_len+len(disease_list), 1000, 0])
    query_len += len(disease_list)

    gene_variant = topic_kb['gene_variant']
    for gene, variant in gene_variant.items():
        gene_list = gene.split(' ')
        query.extend(gene_list)
        q.update([gene.lower()])
        i += 1
        query_id_dict[gene] = i
        q_term_nodes.append([i, query_len, query_len+len(gene_list), 1000, 0])
        query_len += len(gene_list)

        if variant:
            i += 1
            query_id_dict[variant] = i
            variant_list = variant.split(' ')
            query.extend(variant_list)
            q.update([variant.lower()])
            q_term_nodes.append([i, query_len, query_len+len(variant_list), 1000, 0])
            query_len += len(variant_list)

    if 'treatment' in topic_kb:
        q.update([topic_kb['treatment'].lower()])
        i += 1
        treatment = topic_kb['treatment'].lower()
        query_id_dict[treatment] = i
        treatment_list = treatment.split(' ')
        query.extend(treatment_list)
        q_term_nodes.append([i, query_len, query_len + len(treatment_list), 1000, 0])
        query_len += len(treatment_list)

    cur_q_idx = query_id_dict[disease]
    disease_kb = set()
    disease_kb.update([disease])
    if topic_kb['disease_synonyms']:
        disease_kb.update(topic_kb['disease_synonyms'])
    if topic_kb['disease_preferredTerm']:
        disease_kb.update([topic_kb['disease_preferredTerm']])
    if topic_kb['disease_drug_int']:
        disease_kb.update(topic_kb['disease_drug_int'])

    disease_kb = [d_kb for d_kb in disease_kb if d_kb != ' ']

    q_doc_dict[cur_q_idx] = []
    for d_kb in disease_kb:
        d_kb_list = list(map(lambda x: x.lemma_.lower(), nlp(d_kb)))
        if len(d_kb_list) == 1:
            d_kb_list = [d_kb.lower()]
        if ' '.join(d_kb_list) in doc_term_dict:
            q_doc_dict[cur_q_idx].extend(doc_term_dict[' '.join(d_kb_list)])
        else:
            starts, ends, sen_loc = locate_term(d_kb_list, doc_lemma_list, sen_len)
            if starts:
                doc_term_dict[' '.join(d_kb_list)] = []
                q.update([d_kb.lower()])
                for start, end, s_loc in zip(starts, ends, sen_loc):
                    i += 1
                    doc_term_dict[' '.join(d_kb_list)].append(i)
                    doc_term_nodes.append([i, start+query_len, end+query_len, s_loc, 1])

                # print(' '.join(d_kb_list))
                # print(doc_term_dict[' '.join(d_kb_list)])
                q_doc_dict[cur_q_idx].extend(doc_term_dict[' '.join(d_kb_list)])

    q_doc_dict[cur_q_idx] = list(set(q_doc_dict[cur_q_idx]))
    # Gene Term and Variant Term [id, start, end, sen_loc, node_type_id]

    for gene, variant in gene_variant.items():
        cur_q_idx = query_id_dict[gene]
        gene_kb = set()
        gene_kb.update([gene.lower()])
        if gene in topic_kb['gene_alias']:
            gene_kb.update(topic_kb['gene_alias'][gene])
        if gene in topic_kb['gene_official_name']:
            gene_kb.update([topic_kb['gene_official_name'][gene]])
        if gene in topic_kb['gene_drug_int']:
            gene_kb.update(topic_kb['gene_drug_int'][gene])
        gene_kb = [g_kb for g_kb in gene_kb if g_kb != ' ']

        q_doc_dict[cur_q_idx] = []
        for g_kb in gene_kb:
            g_kb_list = list(map(lambda x: x.lemma_.lower(), nlp(g_kb)))
            if len(g_kb_list) == 1:
                g_kb_list = [g_kb.lower()]
            if ' '.join(g_kb_list) in doc_term_dict:
                q_doc_dict[cur_q_idx].extend(doc_term_dict[' '.join(g_kb_list)])
            else:
                starts, ends, sen_loc = locate_term(g_kb_list, doc_lemma_list, sen_len)
                if starts:
                    doc_term_dict[' '.join(g_kb_list)] = []
                    q.update([g_kb.lower()])
                    for start, end, s_loc in zip(starts, ends, sen_loc):
                        i += 1
                        doc_term_dict[' '.join(g_kb_list)].append(i)
                        doc_term_nodes.append([i, start+query_len, end+query_len, s_loc, 1])

                    q_doc_dict[cur_q_idx].extend(doc_term_dict[' '.join(g_kb_list)])
        q_doc_dict[cur_q_idx] = list(set(q_doc_dict[cur_q_idx]))

        if variant:
            cur_q_idx = query_id_dict[variant]
            variant_kb = set()
            variant_kb.update([variant.lower()])
            if topic_kb['variant_drug_int'][variant]:
                variant_kb.update(topic_kb['variant_drug_int'][variant])
            variant_kb = [v_kb for v_kb in variant_kb if v_kb != ' ']

            q_doc_dict[cur_q_idx] = []
            for v_kb in variant_kb:
                v_kb_list = list(map(lambda x: x.lemma_.lower(), nlp(v_kb)))
                if len(v_kb_list) == 1:
                    v_kb_list = [v_kb.lower()]
                if ' '.join(v_kb_list) in doc_term_dict:
                    q_doc_dict[cur_q_idx].extend(doc_term_dict[' '.join(v_kb_list)])
                else:
                    starts, ends, sen_loc = locate_term(v_kb_list, doc_lemma_list, sen_len)
                    if starts:
                        doc_term_dict[' '.join(v_kb_list)] = []
                        q.update([v_kb.lower()])
                        for start, end, s_loc in zip(starts, ends, sen_loc):
                            i += 1
                            doc_term_dict[' '.join(v_kb_list)].append(i)
                            doc_term_nodes.append([i, start+query_len, end+query_len, s_loc, 1])

                        q_doc_dict[cur_q_idx].extend(doc_term_dict[' '.join(v_kb_list)])
            q_doc_dict[cur_q_idx] = list(set(q_doc_dict[cur_q_idx]))

    if 'treatment' in topic_kb:
        treatment = topic_kb['treatment'].lower()
        cur_q_idx = query_id_dict[treatment]

        treatment_kb = set()
        treatment_kb.update([treatment])

        q_doc_dict[cur_q_idx] = []
        for t_kb in treatment_kb:
            t_kb_list = [t_kb]

            if ' '.join(t_kb_list) in doc_term_dict:
                q_doc_dict[cur_q_idx].extend(doc_term_dict[' '.join(t_kb_list)])
            else:
                starts, ends, sen_loc = locate_term(t_kb_list, doc_lemma_list, sen_len)
                if starts:
                    doc_term_dict[' '.join(t_kb_list)] = []
                    q.update([t_kb.lower()])
                    for start, end, s_loc in zip(starts, ends, sen_loc):
                        i += 1
                        doc_term_dict[' '.join(t_kb_list)].append(i)
                        doc_term_nodes.append([i, start + query_len, end + query_len, s_loc, 1])

                    # print(' '.join(d_kb_list))
                    # print(doc_term_dict[' '.join(d_kb_list)])
                    q_doc_dict[cur_q_idx].extend(doc_term_dict[' '.join(t_kb_list)])

        q_doc_dict[cur_q_idx] = list(set(q_doc_dict[cur_q_idx]))

    # Sentence Node [id, start, end, loc, node_type_id]
    for s, sentence in enumerate(s_len):
        i += 1
        sen_nodes.append([i, s, s, s, 2])

    # query + doc sentence
    s_len = [query_len] + s_len

    return query, doc_list, q_term_nodes, doc_term_nodes, sen_nodes, q_doc_dict, s_len


def regularization(text):
    punctuation = string.punctuation
    text = text.replace('-', ' - ')
    text = text.replace('/', ' / ')
    text = re.sub(r'[^A-Za-z0-9 {}]+'.format(punctuation), '', text)
    text = ' '.join([w for w in text.split(' ') if w != '']).strip()
    return text


def data_classification(topic_path, data_path, output_path):
    with open(topic_path, 'r') as f:
        topic = json.load(f)

    data = pd.read_csv(data_path, sep='\t', encoding='utf-8')

    nlp = spacy.load('en_core_sci_md')

    graph_sizes = []
    dataset = {}
    f = jsonlines.open(output_path, 'w')
    for index, row in data.iterrows():
        topic_id = row[0]
        doc_id = row[1]
        text = row[2]
        label = row[3]
        topic_kb = topic[str(topic_id)]
        text = regularization(text)

        query, doc, q_term_nodes, doc_term_nodes, sen_nodes, q_doc_dict, sen_len = preprocessing(text, topic_kb, nlp)

        graph_sizes.append(len(q_term_nodes) + len(doc_term_nodes) + len(sen_nodes))

        if index % 500 == 0:
            print('index: ', index)
            graph = np.array(graph_sizes)
            print(graph.max(-1))
            print(graph.mean(-1))

        dataset['query'] = query
        dataset['doc'] = doc
        if label != 0:
            label = 1
        dataset['label'] = label
        dataset['query_id'] = str(topic_id)
        dataset['doc_id'] = str(doc_id)
        dataset['q_term_nodes'] = q_term_nodes
        dataset['doc_term_nodes'] = doc_term_nodes
        dataset['sen_nodes'] = sen_nodes
        dataset['q_doc_dict'] = q_doc_dict
        dataset['sen_len'] = sen_len
        dataset['retrieval_score'] = 0.0
        f.write(dataset)

    # graph_sizes = np.array(graph_sizes)
    # print(graph_sizes.max(-1))
    # print(graph_sizes.mean(-1))
    f.close()


def data_ranking(topic_path, data_path, output_path):
    np.random.seed(2021)
    data = pd.read_csv(data_path, sep='\t', encoding='utf-8')
    with open(topic_path, 'r') as f:
        topic = json.load(f)

    nlp = spacy.load('en_core_sci_md')

    f = jsonlines.open(output_path, 'w')

    num = 0
    for topic_id, topic_kb in topic.items():
        if int(topic_id) not in [28, 29, 30, 76, 77, 78, 79, 80]:

            data_query = data.loc[data['topic'] == int(topic_id)].reset_index(drop=True)
            data_label_0 = data_query.loc[data_query['label'] == 0].sample(frac=0.1).reset_index(drop=True)
            data_label_1 = data_query.loc[data_query['label'] == 1].reset_index(drop=True)
            data_label_2 = data_query.loc[data_query['label'] == 2].reset_index(drop=True)
            data_label_2 = pd.concat([data_label_2, data_label_1], axis=0).sample(frac=0.5).reset_index(drop=True)

            for idx, positive in data_label_2.iterrows():
                dataset = {}
                text_pos = positive[2]
                text_pos = regularization(text_pos)
                query, doc_pos, q_term_nodes_pos, doc_term_nodes_pos, sen_nodes_pos, q_doc_dict_pos, sen_len_pos = \
                    preprocessing(text_pos, topic_kb, nlp)
                for i, negative in data_label_0.iterrows():
                    text_neg = negative[2]
                    text_neg = regularization(text_neg)

                    _, doc_neg, q_term_nodes_neg, doc_term_nodes_neg, sen_nodes_neg, q_doc_dict_neg, sen_len_neg = \
                        preprocessing(text_neg, topic_kb, nlp)

                    dataset['query'] = query
                    dataset['doc_pos'] = doc_pos
                    dataset['doc_neg'] = doc_neg
                    dataset['q_term_nodes_pos'] = q_term_nodes_pos
                    dataset['doc_term_nodes_pos'] = doc_term_nodes_pos
                    dataset['sen_nodes_pos'] = sen_nodes_pos
                    dataset['q_doc_dict_pos'] = q_doc_dict_pos
                    dataset['sen_len_pos'] = sen_len_pos
                    dataset['q_term_nodes_neg'] = q_term_nodes_neg
                    dataset['doc_term_nodes_neg'] = doc_term_nodes_neg
                    dataset['sen_nodes_neg'] = sen_nodes_neg
                    dataset['q_doc_dict_neg'] = q_doc_dict_neg
                    dataset['sen_len_neg'] = sen_len_neg
                    f.write(dataset)

                    if num % 1000 == 0:
                        print('num:', num)
                    num += 1
    f.close()


def data_ranking_2(topic_path, ranking_data_path, output):
    with jsonlines.open(ranking_data_path, 'r') as f, jsonlines.open(output, 'w') as o, open(topic_path, 'r') as t:
        topic = json.load(t)

        nlp = spacy.load('en_core_sci_md')
        num = 0
        for line in f:
            query_id = line['query_id']
            text_pos = line['doc_pos']
            text_pos = regularization(text_pos)
            text_neg = line['doc_neg']
            text_neg = regularization(text_neg)

            topic_kb = topic[str(query_id)]

            dataset = {}

            query, doc_pos, q_term_nodes_pos, doc_term_nodes_pos, sen_nodes_pos, q_doc_dict_pos, sen_len_pos = \
                preprocessing(text_pos, topic_kb, nlp)

            query_2, doc_neg, q_term_nodes_neg, doc_term_nodes_neg, sen_nodes_neg, q_doc_dict_neg, sen_len_neg = \
                preprocessing(text_neg, topic_kb, nlp)

            dataset['query'] = query
            dataset['doc_pos'] = doc_pos
            dataset['doc_neg'] = doc_neg
            dataset['q_term_nodes_pos'] = q_term_nodes_pos
            dataset['doc_term_nodes_pos'] = doc_term_nodes_pos
            dataset['sen_nodes_pos'] = sen_nodes_pos
            dataset['q_doc_dict_pos'] = q_doc_dict_pos
            dataset['sen_len_pos'] = sen_len_pos
            dataset['q_term_nodes_neg'] = q_term_nodes_neg
            dataset['doc_term_nodes_neg'] = doc_term_nodes_neg
            dataset['sen_nodes_neg'] = sen_nodes_neg
            dataset['q_doc_dict_neg'] = q_doc_dict_neg
            dataset['sen_len_neg'] = sen_len_neg
            dataset['query_id'] = query_id
            dataset['doc_pos_id'] = line['doc_pos_id']
            dataset['doc_neg_id'] = line['doc_neg_id']
            o.write(dataset)
            if num % 1000 == 0:
                print('num:', num)
            num += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='classification')

    parser.add_argument('-topic_path', type=str, default='./data/train_dev_topic_kb.json')
    parser.add_argument('-data_path', type=str, default='./data/train_text.csv')
    parser.add_argument('-output_path', type=str, default='./data/trec_graph_train.jsonl')

    args = parser.parse_args()

    if args.task == 'classification':
        data_classification(args.topic_path, args.data_path, args.output_path)
    elif args.task == 'ranking':
        # data_ranking(args.topic_path, args.data_path, args.output_path)
        data_ranking_2(args.topic_path, args.data_path, args.output_path)
    else:
        raise ValueError('Task must be `ranking` or `classification`.')


if __name__ == '__main__':
    main()



