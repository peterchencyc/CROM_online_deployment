# Submission 549
<p align="center">
<img src="teaser.png"  width="700"/>
</p>
This repository contains the source code for reproducing Figure 9 in the main text. A pre-trained network is provided.

## Prerequisites
We assume a fresh install of Ubuntu 20.04. For example,

```
docker run --gpus all --shm-size 128G -it --rm -v $HOME:/home/ubuntu ubuntu:20.04
```

Install python and pip:
```
apt-get update
apt install python3-pip
```

## Dependencies
Install python package dependencies through pip:

```
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

## Execute
### Execute Diffusion
```
python3 online/execute_diffusion.py -device [device]
```
### Execute Diffuse Image
```
python3 online/execute_diffuseimage.py -device [device]
```
### Execute Elasticity
```
python3 online/execute_online.py -device [device]
```
[device] is either cpu or cuda

The solutions are stored in the output directory.

## Optimal Sampling
### Execute Diffusion
```
python3 optimal_sampling/execute_diffusion.py -device [device]
```
### Execute Diffuse Image
```
python3 optimal_sampling/execute_diffuseimage.py -device [device]
```

We suggest to use GPU for optimal sampling. Normally it needs hours to get the optimal sampling.

