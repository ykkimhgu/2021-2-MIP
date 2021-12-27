import pandas as pd
import xml.etree.ElementTree as et
import os 
path = '/home/ssslab/Desktop/Dataset/cctv_data/cubox_cctv_data/violence'
Data=[]
print("start change")
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    xtree = et.parse(fullname)
    xroot=xtree.getroot()
    
    for node in xroot:

      file_name=node.find("filename").text
      a=node.find("object")
      action=a.find("action")
      frame=action.find("frame")
      try:
        end_frame=frame.find("end").text
        start_frame=frame.find("start").text
      except Exception as e:
        continue
      Data.append({"file_name":file_name,"start_frame":int(start_frame)-5,"end_frame":int(end_frame)+5})

out_df=pd.DataFrame(Data)
out_df.to_csv("./file/violence_name.csv")
print("clear")

