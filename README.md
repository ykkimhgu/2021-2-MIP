# Violence and Fall Detection using Action Classification for Smart CCTV

<img src="demo.gif" alt="demo" style="zoom:50%;" />

## Introduction

This repository contains MoViNet-pytorch (https://github.com/Atze00/MoViNet-pytorch) and Violence and Fall Detection code. It is a program that detect the Specific abnormal behavior using Movinet . It is used Airport Abnormal behavior CCTV dataset from AI-Hub. 

This project was carried out as a 2021-2 MIP study with advisor Ph.D Young-Keun Kim

## Description

The implementation is based on two papers:

-  [Movinets: Mobile video networks for efficient video recognition](http://openaccess.thecvf.com/content/CVPR2021/html/Kondratyuk_MoViNets_Mobile_Video_Networks_for_Efficient_Video_Recognition_CVPR_2021_paper.html)
- MoVinet-Pytorch https://github.com/Atze00/MoViNet-pytorch

## Requirements

- Python >=3.8 
- Numpy `pip install numpy`
- PIL `pip3 install pillow`
- OpenCV `pip3 install opencv-python`
- av `pip install av`
- PyTorch
- torchvision 
- torchaudio 

## Tutorial 

##### GitHub

1. Go to the https://github.com/hkim1207/2021MIP
2. Download all the files in your local PC

##### Anaconda

1. open the Anaconda Prompt (anaconda3)
   
2. Change the current location to the directory where the Python file to be executed is in.

   `cd Desktop/conda/Path` 

3. Create a new conda environment

   `conda create -n MoViNet python=3.8` 

4. Activate the MoVinet environment

   `conda activate MoViNet`

5. Install pytorch and torch family. 

   `conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch`

6. Install the other libraries to run the train&test.py

   `pip install -U -r requirements.txt`

## Train & Test

`train_test.py` 

```bash
python Train_test.py
```



`video_fall.py`

```bash
cd video_test
video_fall.py
```

### Citations

```
@article{kondratyuk2021movinets,
  title={MoViNets: Mobile Video Networks for Efficient Video Recognition},
  author={Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang, Matthew Brown, and Boqing Gong},
  journal={arXiv preprint arXiv:2103.11511},
  year={2021}
}

https://github.com/Atze00/MoViNet-pytorch
```

## Info

If you have a question about the code or setting the environment, contact to me via e-mail

21701052@handong.edu
