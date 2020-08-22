# AGE
Source code and datasets for KDD 2020 paper "Adaptive Graph Encoder for Attributed Graph Embedding"

---

## Requirements

Please make sure your environment includes:

```
python (tested on 3.7.4)
pytorch (tested on 1.2.1)
```

Then, run the command:
```
pip install -r requirements.txt
```

## Run

Run AGE on Cora dataset:

```
python train.py --dataset cora --gnnlayers 8 --upth_st 0.011 --lowth_st 0.1 --upth_ed 0.001 --lowth_ed 0.5
```

To reproduce the node clustering experiment results, please follow our hyper-parameter settings:

| Dataset  | gnnlayers | upth_st | lowth_st | upth_ed | lowth_ed |
| :------- | --------- | ------- | -------- | ------- | -------- |
| Cora     | 8         | 0.0110  | 0.0010   | 0.1     | 0.5      |
| Citeseer | 3         | 0.0015  | 0.0010   | 0.1     | 0.5      |
| Wiki     | 1         | 0.0011  | 0.0010   | 0.1     | 0.5      |
| Pubmed   | 35        | 0.0013  | 0.0010   | 0.7     | 0.8      |

For link prediction, please run `link_pred.py`. We did not tune hyper-parameters for link prediction, so you can tune all kinds of hyper-parameters to get better performance.

## Cite

If you use the code, please cite our paper:

```
@inproceedings{cui2020adaptive,
  title={Adaptive Graph Encoder for Attributed Graph Embedding},
  author={Cui, Ganqu and Zhou, Jie and Yang, Cheng and Liu, Zhiyuan},
  booktitle={Proceedings of SIGKDD 2020},
  year={2020}
}
```