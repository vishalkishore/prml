import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from loguru import logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(data_set, batch_size=32, shuffle=False)

class cnn_feature_ext(nn.Module):
    def __init__(self):
        super(cnn_feature_ext, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
    def forward(self,x):
        x=self.conv(x)
        return x
model = cnn_feature_ext()
model.eval()
def feature_ext(dataloader,model):
    feature_list=[]
    with torch.no_grad():
        for images,temp in dataloader:
            feature=model(images)
            feature_list.append(feature.view(feature.size(0),-1))
    return torch.cat(feature_list)
cnn_feature = feature_ext(dataloader, model)
torch.save(cnn_feature,"feature_using_cnn.pt")
logger.info("done")
logger.critical(cnn_feature.shape)