# Learning Topological Representations with Bidirectional Graph Attention Network for Solving Job Shop Scheduling Problem

Paper: https://arxiv.org/abs/2402.17606

If you make use of the code/experiment or TBGAT algorithm in your work, please cite our paper (Bibtex below).
```

@InProceedings{zhanglearning2024,
  title = 	 {Learning Topological Representations with Bidirectional Graph Attention Network for Solving Job Shop Scheduling Problem},
  author =       {Zhang, Cong and Cao, Zhiguang and Wu, Yaoxin and Song, Wen and Sun, Jing},
  booktitle = 	 {Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence},
  year = 	 {2024},
  editor = 	 {Evans, Robin J. and Shpitser, Ilya},
  volume = 	 {},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {},
  publisher =    {PMLR},
  pdf = 	 {},
  url = 	 {},
}

```


### Manual Setup
python 3.9.x

cuda 10.2 + torch 1.10.0
```commandline
pip3 install torch==1.10.0 torchvision==0.11.0 torchaudio===0.10.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
Install dependencies:
```commandline
pip install --upgrade pip
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html
pip install torch-geometric==2.0.3
pip install matplotlib==3.4.3
pip install ortools==9.3.10497
pip install openpyxl
```

### Docker Setup
Clone this repo and within the repo folder run the following command.

Create image `neural-tabu-jssp-image`:
```commandline
docker build -t neural-tabu-jssp-image .
```

Create container `neural-tabu-jssp-container` from `neural-tabu-jssp-image`, and activate it:
```commandline
docker run --gpus all --name neural-tabu-jssp-container -it neural-tabu-jssp-image
```
