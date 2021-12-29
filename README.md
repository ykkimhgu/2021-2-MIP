# Violence and Fall Detection using Action Classification for Smart CCTV

<img src="demo.gif" alt="demo" style="zoom:50%;" />

## Introduction

This repository contains MoViNet-pytorch (https://github.com/Atze00/MoViNet-pytorch) and Violence and Fall Detection code. It is a program that detect the Specific abnormal behavior using Movinet . It is used Airport Abnormal behavior CCTV dataset from AI-Hub. 

This project was carried out as a 2021-2 MIP study with advisor Ph.D Young-Keun Kim

## Description

The implementation is based on two papers:

-  [Movinets: Mobile video networks for efficient video recognition](http://openaccess.thecvf.com/content/CVPR2021/html/Kondratyuk_MoViNets_Mobile_Video_Networks_for_Efficient_Video_Recognition_CVPR_2021_paper.html)
-  MoVinet-Pytorch https://github.com/Atze00/MoViNet-pytorch

## Requirements

- U
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

### Load library

```aasfaf
import torch
import torchvision
import datasets
import transforms
import train
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T
from movinets import MoViNet
from movinets.config import _C
```

### dataset load

```bash
dataset_train = datasets.VideoLabelDataset(
	"file/class2_train_faint.csv",# train path(csv file)
    transform=torchvision.transforms.Compose([
        transforms.VideoFilePathToTensor(max_len=16, fps=30, padding_mode='last'),
        
        transforms.VideoRandomCrop([172, 172]),
        transforms.VideoResize([172, 172]),
    ])
)

dataset_test = datasets.VideoLabelDataset(
    "file/class2_test_faint.csv",# test path (csv file)
    transform=torchvision.transforms.Compose([
        transforms.VideoFilePathToTensor(max_len=16, fps=30, padding_mode='last'),
        transforms.VideoRandomCrop([172, 172]),
        transforms.VideoResize([172, 172]),
    ])
)

data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 16, shuffle = True)


data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 16, shuffle = False)

```

### load model

```torch.cuda.empty_cache()
model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True )
start_time = time.time()

trloss_val, tsloss_val = [], []
class_number = 2  # change here if you want to other class number
model.classifier[3] = torch.nn.Conv3d(2048,class_number, (1,1,1)) # class number
```

### train and evaluate

```print("start ")
optimz = optim.Adam(model.parameters(), lr=0.00005)
for epoch in range(1, 9):
    print('Epoch:', epoch)
    train.train_iter(model, optimz, data_loader_train, trloss_val)
    #model save
    #torch.save(model.state_dict(),'./model_data/class4/Movinet_class4_preY_last_{}.pth'.format(epoch+1))
    train.evaluate(model,data_loader_test, tsloss_val)#model evauate
```

## demo

`video_fall.py`

### load library

```
import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
import transforms_mod as tt
import torchvision
import transforms
from movinets import MoViNet
from movinets.config import _C
import PIL
```

### model load

```
filePath='your_path.mp4'


model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True )
model.classifier[3] = torch.nn.Conv3d(2048, 2, (1,1,1))

#model load
model.load_state_dict(torch.load('./model_data/class2/Movinet_class2_pre_faint_5.pth'))

model.cuda()
model.eval()

model.clean_activation_buffers()
```

### predict video

```
frame_count = 0

org=(550,150)
font=cv2.FONT_HERSHEY_SIMPLEX 
 
flag=0
cap = cv2.VideoCapture(filePath)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out = cv2.VideoWriter('last3.avi', fourcc, 30.0, (int(width), int(height)))
fps=30
normal_cnt=0
Fall_cnt=0
a_cnt=0
pre_frame=0
cur_frame=0
max_len=16
channels=3
padding_mode='last'
n=16
text=''
ff = torch.FloatTensor(3, 500, height, width)
fa = torch.FloatTensor(3, 500, height, width)
text_c=(255,0,0)
text="Normal"
frames = torch.FloatTensor(3, n, height, width)
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # print(frame.shape)
        frame_org=frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frame = torch.from_numpy(frame)
    #         # (H x W x C) to (C x H x W)
        frame = frame.permute(2, 0, 1)
        frames[:,frame_count%n, :, :] = frame.float()
        last_idx=frame_count
        start_idx=last_idx-16
        ff[:,frame_count, :, :] = frame.float()
        
        if frame_count>16:
            fa=ff[:,start_idx:last_idx, :, :]

        frame_count=frame_count+1



        if frame_count%(n-1)==0:

            w=172
            h=172

            C, L, H, W = frames.size()

            rescaled_video = torch.FloatTensor(C, L, h, w)
                    
                    # use torchvision implemention to resize video frames
            transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.Resize((172,172), PIL.Image.BILINEAR),
                        torchvision.transforms.ToTensor(),
                    ])

            for l in range(L):
                frame = rescaled_video[:, (l), :, :]
                frame = transform(frame)
                rescaled_video[:, l, :, :] = frame
            #rescaled_video[:,,:,:]
            rescaled_vide = rescaled_video[:, -1:-1*(n-1), :, :]


            rescaled_video=rescaled_video.reshape(1,3,n,172,172)

            output= F.log_softmax(model(rescaled_video.cuda()), dim=1) 
            _, pred = torch.max(output, dim=1) 
            print(pred)
            if (pred==1): #Fall
                #text="Fall"
                Fall_cnt=Fall_cnt+1
                cur_frame=1

            elif (pred==0 & flag==0): #normal
                text="Normal"
                normal_cnt=normal_cnt+1
                cur_frame=0
                text_c=(255,0,0)

            if(Fall_cnt>5 ):    

                text="Fall"
                text_c=(0,0,255)
                a_cnt=a_cnt+1
                # Fall_cnt=0
                if (a_cnt>10):

                    print("aaaaa")
                    text="normal" 
                    text_c=(255,0,0)
                   
                    Fall_cnt=0
            pre_frame=cur_frame

    
    cv2.putText(frame_org,str(text),org,font,3,text_c,8)
    cv2.imshow("image",frame_org)
    #out.write(frame_org) 
    if cv2.waitKey(33) ==ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
```

### run 

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
