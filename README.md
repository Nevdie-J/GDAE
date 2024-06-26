# Requirements

+ CUDA 11.1
+ torch==1.9.1+cu111
+ torch_geometric==2.1.0
+ torch-cluster==1.5.9
+ torch-scatter==2.0.7
+ torch-sparse==0.6.10

# Installation

```bash
pip install -r requirements.txt
```

# Running
## Performance on Synthetic Graphs
+ node classification
```bash
python main_gdae.py -dataset attribute -task nc -lam1 0.7 -norm
python main_gdae.py -dataset tl-40 -task nc -lam1 0.7 -norm
python main_gdae.py -dataset tl-60 -task nc -lam1 0.7 -norm
python main_gdae.py -dataset tl-80 -task nc -lam1 0.7 -norm
python main_gdae.py -dataset topology -task nc -lam1 0.7 -norm
```

## Performance on Real-world Datasets
+ node classification
```bash
python main_gdae.py -dataset cora -task nc
```
+ link prediction
```bash
python main_gdae.py -dataset cora -task lp
```