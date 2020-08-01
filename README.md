# AGE
Source code and dataset for KDD 2020 paper "Adaptive Graph Encoder for Attributed Graph Embedding"

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