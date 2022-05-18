# Knowledge enhanced edge-driven graph neural ranking (KEGNR)




\* Training


```
sh train_graph.sh
```
\* Inference

```
sh inference_graph.sh
```


\* Evaluation

```
./trec_eval.9.0 -m all_trec ./data/qrel.txt ./results/result.txt
```
