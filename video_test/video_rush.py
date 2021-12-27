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
# 
filePath='/home/ssslab/Downloads/videos/A_faint_1.mp4'

print(filePath)

model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True )
model.classifier[3] = torch.nn.Conv3d(2048, 2, (1,1,1))

model.load_state_dict(torch.load('./model_data/class2/Movinet_class2_preno_faint_9.pth'))

# video_capture = cv2.VideoCapture(filePath)

model.cuda()
model.eval()

model.clean_activation_buffers()

frame_count = 0

org=(25,50)
font=cv2.FONT_HERSHEY_SIMPLEX 
 
 
z=[]
p=[]
cap = cv2.VideoCapture(filePath)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')


height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out = cv2.VideoWriter('A_normal_1R.avi', fourcc, 30.0, (int(width), int(height)))
fps=30
normal_cnt=0
rush_cnt=0

max_len=16
channels=3
padding_mode='last'
n=16
text=''
ff = torch.FloatTensor(3, 500, height, width)
fa = torch.FloatTensor(3, 500, height, width)
text_c=(0,0,0)
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
            # ff=
            # ff=ff/255
            # fa=fa/255
            # frame/255
            # frames /= 255
            w=172
            h=172
            # h, w = self.size
            # print(frames.shape)
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

            #frames = transform(frames)
            rescaled_video=rescaled_video.reshape(1,3,n,172,172)
            #print(rescaled_video)
            #ff=ff[:,-16::,:,:]
            output= F.log_softmax(model(rescaled_video.cuda()), dim=1) 
            _, pred = torch.max(output, dim=1) 

            if (pred==1):
                text="normal"
                normal_cnt=normal_cnt+1
                text_c=(255,0,0)
                cv2.putText(frame_org,str(text),org,font,2,text_c,2)
    
            elif (pred==0):
                # text="rush"
                rush_cnt=rush_cnt+1
                # text_c=(0,0,255)
                # cv2.putText(frame_org,str(text),org,font,2,text_c,2)
    if(rush_cnt>3):
        text="rush"
        text_c=(0,0,255)
        rush_cnt=0
                        
    
            #frame_count=0
    cv2.putText(frame_org,str(text),org,font,2,text_c,2)
    cv2.imshow("image",frame_org)
    # out.write(frame_org) 
    if cv2.waitKey(66) ==ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
            # z=[]
print(normal_cnt,rush_cnt)
# cv2.waitKey(3)




############################
'''

with torch.no_grad():
    while video_capture.isOpened():

  
        ret, frame = video_capture.read()

  # frame = cv2.resize(frame, (172, 172), interpolation=cv2.INTER_AREA)
        if not ret:

            video_capture.release()

            break
        else:
            frame = cv2.resize(frame, (172, 172), interpolation=cv2.INTER_AREA)
        frame_count += 1
        frame_org = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame=frame_org.transpose()
        z.append(frame)
        if (frame_count % 8) == 0:
        
            video=np.array(z) # weight of size [8, 3, 1, 3, 3]
            rescaled_video= video.reshape(1,3,8,172,172) # N, C, T,W,H
            rescaled_video = torch.from_numpy(rescaled_video)
            rescaled_video=rescaled_video/255
    # print(rescaled_video)
            output= F.log_softmax(model(rescaled_video.cuda()), dim=1) 
            _, pred = torch.max(output, dim=1) 
            p.append(pred[0])
            text=""
            if (pred==1):
                normal_cnt=normal_cnt+1
                text="Normal"
            elif (pred==0):
                text="rush"
                rush_cnt=rush_cnt+1
            cv2.putText(frame_org,str(text),org,font,1,(255,0,0),2)
            cv2.imshow("image",frame_org) 
            z=[]

            cv2.waitKey(3)









#     video=torchvision.io.read_video(filePath,0,7/30,pts_unit="sec")
#     video = torch.from_numpy(np.asarray(video[0]))
#     # # (H x W x C) to (C x H x W)
#     ## L WH C -> C L W H
#     video = video.permute(0, 3, 1, 2)
#     #video= video.reshape(1,8,3,1080,1920)
#     # video = video.reshape()
#     # frames[:, index, :, :] = video.float()
#     # print(type(video))
#     video=torch.tensor(video)
#     # print(video.shape)
#     L, C, W, H = video.size()
#     rescaled_video = torch.FloatTensor(L, C, 172, 172)
    
#     # use torchvision implemention to resize video frames
#     transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToPILImage(),
#         torchvision.transforms.Resize((172,172), PIL.Image.BILINEAR),
#         torchvision.transforms.ToTensor(),
#     ])

#     for l in range(L):
#         frame = video[l, :, :, :]
#         # print(frame.shape)
#         frame = transform(frame)
#         rescaled_video[l,:,:, :] = frame

#     #print(rescaled_video.shape)
#     rescaled_video= rescaled_video.reshape(8,3,1,172,172)
#     # print(rescaled_video)
#     output= F.log_softmax(model(rescaled_video.cuda()), dim=1)
#     _, pred = torch.max(output, dim=1)
            
#     print(pred)

# # while True:
# #     # 한 장의 이미지(frame)를 가져오기
# #     # 영상 : 이미지(프레임)의 연속
# #     # 정상적으로 읽어왔는지 -> retval
# #     # 읽어온 프레임 -> frame
# #     retval, frame = cap.read()
    
# #     # print(frame.shape)
# #     frame = cv2.resize(frame, (172, 172), interpolation=cv2.INTER_AREA)
# #     # print(frame.shape)

# #     cv2.putText(frame,text,org,font,1,(255,0,0),2)

    

# #     if (frame_cnt%8 ==0):
# #         # print("zz")
# #         cv2.imshow('frame', frame)  # 프레임 보여주기


# #     # frame=frame.reshape(1,172,172,3)
# #     # print(frame.shape)
# #     # # print(type(frame))
# #     # z=tt.resize(frame,172)
# #     # z=torch.Tensor(frame)
# #     # print(z.shape)
# #     # output = F.log_softmax(model(z.cuda()), dim=1)
# #     frame_cnt=frame_cnt+1

# #     if not(retval):   # 프레임정보를 정상적으로 읽지 못하면
# #         break  # while문을 빠져나가기
        
# #     key = cv2.waitKey(30)  # frameRate msec동안 한 프레임을 보여준다
    
# #     # 키 입력을 받으면 키값을 key로 저장 -> esc == 27(아스키코드)
# #     if key == 27:
# #         break   # while문을 빠져나가기
        
# # if cap.isOpened():   # 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
# #     cap.release()   # 영상 파일(카메라) 사용을 종료
    
# # cv2.destroyAllWindows()
'''