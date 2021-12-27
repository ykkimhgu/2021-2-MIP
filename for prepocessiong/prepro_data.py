import torchvision

import pandas as pd
path="/home/ssslab/Desktop/Dataset/cctv_data/cubox_cctv_data/violence/"

data_csv=pd.read_csv("./file/violence_name.csv")


Data=[]
for i in range(len(data_csv)):

  file_name=data_csv["file_name"][i]
  Data.append({"file_name":"./data_clip_violence/"+file_name,"label":2})

  # print("idx: ",i)
  name=data_csv['file_name'][i]
  start=data_csv['start_frame'][i]/30
  end=data_csv["end_frame"][i]/30
  video=torchvision.io.read_video(path+name,start,end,pts_unit="sec")
  torchvision.io.write_video("./data_clip_violence/"+"{}".format(name),video[0],30)

out_df=pd.DataFrame(Data)
out_df.to_csv("./file/violence_train.csv")
