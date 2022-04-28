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
pip install ortools==9.0.9048
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