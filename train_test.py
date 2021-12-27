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

dataset_train = datasets.VideoLabelDataset(
	"file/class2_train_faint.csv",
    transform=torchvision.transforms.Compose([
        transforms.VideoFilePathToTensor(max_len=16, fps=30, padding_mode='last'),
        
        transforms.VideoRandomCrop([172, 172]),
        transforms.VideoResize([172, 172]),
    ])
)

dataset_test = datasets.VideoLabelDataset(
    "file/class2_test_faint.csv",
    transform=torchvision.transforms.Compose([
        transforms.VideoFilePathToTensor(max_len=16, fps=30, padding_mode='last'),
        transforms.VideoRandomCrop([172, 172]),
        transforms.VideoResize([172, 172]),
    ])
)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 16, shuffle = True)


data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 16, shuffle = False)


torch.cuda.empty_cache()
model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True )
start_time = time.time()

trloss_val, tsloss_val = [], []
#class number = 2  
model.classifier[3] = torch.nn.Conv3d(2048, 2, (1,1,1)) # class number

#model load
#model.load_state_dict(torch.load('./model_data/class4/Movinet_class4_preY_last_9.pth'))

# print(model)

##train
print("start ")
optimz = optim.Adam(model.parameters(), lr=0.00005)
for epoch in range(1, 9):
    print('Epoch:', epoch)

    train.train_iter(model, optimz, data_loader_train, trloss_val)
    #model save
    #torch.save(model.state_dict(),'./model_data/class4/Movinet_class4_preY_last_{}.pth'.format(epoch+1))
    #model evauate
    train.evaluate(model,data_loader_test, tsloss_val)
    
print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


