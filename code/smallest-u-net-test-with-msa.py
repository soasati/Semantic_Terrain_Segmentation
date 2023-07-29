#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Set up configuration

# In[ ]:


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# In[ ]:


#!pip install segmentation-models-pytorch
get_ipython().system('pip install -U git+https://github.com/albumentations-team/albumentations')
get_ipython().system('pip install --upgrade opencv-contrib-python')


# In[ ]:


# !pip install torch==1.11.0


# In[ ]:


import sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import csv


# In[ ]:


print(torch.__version__)


# # Set-up configurations

# In[ ]:


IMAGE_PATH="/kaggle/input/vale-semantic-terrain-segmentation/raw_images/raw_images/"
MASK_PATH="/kaggle/input/vale-semantic-terrain-segmentation/mask_rgb_filled/mask_rgb_filled/"
DEVICE=torch.device('cuda')
EPOCHS=100
LR=0.0000025
BATCH_SIZE=15
LOAD_MODEL=False


# # Bad images

# In[ ]:


bad_images=[]
"""
for test_image in os.listdir(IMAGE_PATH):
    try:
        image=cv2.imread(IMAGE_PATH+test_image)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    except Exception as err:
        print(err)
        bad_images.append(test_image)
"""
bad_images.append("05200.png")
print(bad_images)


# # Convert segmentation masks into one-hot-encoding.

# In[ ]:


def get_one_hot_encoding(mask):
    #Yellow matrix
    YELLOW = np.array([255, 255, 0])
    YELLOW_Matrix = np.zeros((1080, 1920))
    yellow_pixels = np.all(mask == YELLOW, axis=2)
    YELLOW_Matrix[yellow_pixels] = 1

    #Red matrix
    RED=np.array([255,0,0])
    RED_Matrix=np.zeros((1080,1920))
    red_pixels=np.all(mask== RED,axis=2)
    RED_Matrix[red_pixels]=1

    #Green matrix
    GREEN=np.array([0,255,0])
    GREEN_Matrix=np.zeros((1080,1920))
    green_pixels=np.all(mask== GREEN,axis=2)
    GREEN_Matrix[green_pixels]=1

    #Orange matrix
    ORANGE=np.array([255,128,0])
    ORANGE_Matrix=np.zeros((1080,1920))
    orange_pixels=np.all(mask== ORANGE,axis=2)
    ORANGE_Matrix[orange_pixels]=1

    stacked=np.stack((YELLOW_Matrix,RED_Matrix,GREEN_Matrix,ORANGE_Matrix),axis=2)
    return stacked


# In[ ]:


def one_hot_decoding(one_hot_encoded):
    # Prepare the blank image with the same dimensions as the input
    decoded_image = np.zeros((one_hot_encoded.shape[0], one_hot_encoded.shape[1], 3), dtype=np.uint8)

    # Yellow
    YELLOW = np.array([255, 255, 0])
    yellow_pixels = one_hot_encoded[:, :, 0] == 1
    decoded_image[yellow_pixels] = YELLOW

    # Red
    RED = np.array([255, 0, 0])
    red_pixels = one_hot_encoded[:, :, 1] == 1
    decoded_image[red_pixels] = RED

    # Green
    GREEN = np.array([0, 255, 0])
    green_pixels = one_hot_encoded[:, :, 2] == 1
    decoded_image[green_pixels] = GREEN

    # Orange
    ORANGE = np.array([255, 128, 0])
    orange_pixels = one_hot_encoded[:, :, 3] == 1
    decoded_image[orange_pixels] = ORANGE

    return decoded_image


# # Split into sixteen and reconstruct

# In[ ]:


def split_into_sixteen(image,mask, rows=4, cols=4):
    # Get the dimensions of the input image
    if(len(image.shape)==3): #Loaded using cv2
        height, width, _ = image.shape
    else: #Loaded as tensor
        n,c,height,width=image.shape

    # Calculate the size of each block
    block_height = height // rows
    block_width = width // cols

    # Initialize an empty list to store the divided image parts
    divided_image_parts = []
    divided_mask_parts=[]
    # Loop through the image grid and append each part to the divided_parts list
    #print("type(Image)= ",type(image))
    for i in range(rows):
        for j in range(cols):
            # Compute the coordinates of the current block
            y_start = i * block_height
            y_end = (i + 1) * block_height
            x_start = j * block_width
            x_end = (j + 1) * block_width

            # Crop the image and append it to the divided_parts list
            if(isinstance(image,torch.Tensor)):
                #print("Tensor")
                divided_image_parts.append(image[:,:,y_start:y_end,x_start:x_end])
                divided_mask_parts.append(mask[:,:,y_start:y_end,x_start:x_end])
            
            else:
                divided_image_parts.append(image[y_start:y_end, x_start:x_end,:])
                divided_mask_parts.append(mask[y_start:y_end,x_start:x_end,:])

    return divided_image_parts,divided_mask_parts

def reconstruct_image(divided_image_parts, divided_mask_parts,rows=4, cols=4):
    image_rows = []
    mask_rows=[]

    for i in range(rows):
        image_row = np.hstack(divided_image_parts[i * cols : (i + 1) * cols])
        mask_row=np.hstack(divided_mask_parts[i*cols: (i+1)*cols])
        image_rows.append(image_row)
        mask_rows.append(mask_row)

    reconstructed_image = np.vstack(image_rows)
    reconstructed_mask=np.vstack(mask_rows)
    
    return reconstructed_image,reconstructed_mask


# # Augmentations

# In[ ]:


import albumentations as A


# In[ ]:


def get_augs():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=30, max_width=30, min_holes=2, min_height=10, min_width=10, p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    ])


# # CSV FILE AND Pandas Dataframe

# In[ ]:


CSV_FILENAME="file.csv"
with open(CSV_FILENAME,'w') as csvfile:
    fwriter=csv.writer(csvfile,delimiter=',',
                      quotechar=',', quoting=csv.QUOTE_MINIMAL)
    fwriter.writerow(['image_path','mask_path'])
    for name in os.listdir(IMAGE_PATH):
        mask=cv2.imread(MASK_PATH+name)
        mask=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        fwriter.writerow([IMAGE_PATH+name,MASK_PATH+name])


df=pd.read_csv(CSV_FILENAME)
df.head()
print(len(df))


# In[ ]:


#Drop bad images
for bad_image in bad_images:
    df=df.drop(df[df['image_path']==IMAGE_PATH+bad_image].index)
print(len(df))


# # Train-val-test split

# In[ ]:


df_train,df_test=train_test_split(df,test_size=0.2,random_state=42)
df_val,df_test=train_test_split(df_test,test_size=0.5,random_state=42)
print("size of df_train= ",len(df_train))
print("size of df_val= ",len(df_val))
print("size of df_test= ", len(df_test))


# In[ ]:


df_train.head()


# In[ ]:


df_val.head()


# In[ ]:


df_test.head()


# In[ ]:


image=df_test.loc[24,'image_path']
print(image)
mask=df_test.loc[24,'mask_path']
print(mask)


# In[ ]:


from PIL import Image


# In[ ]:


# Load the images
image_path = "/kaggle/input/vale-semantic-terrain-segmentation/raw_images/raw_images/05077.png"
mask_path = "/kaggle/input/vale-semantic-terrain-segmentation/mask_rgb_filled/mask_rgb_filled/05077.png"

# Open and display the images
image = Image.open(image_path)
mask = Image.open(mask_path)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the image on the left subplot
axs[0].imshow(image)
axs[0].set_title("Image")

# Plot the mask on the right subplot
axs[1].imshow(mask)
axs[1].set_title("Mask")

# Remove the axis labels
for ax in axs:
    ax.axis("off")

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()


# # Custom Dataset

# In[ ]:


from torch.utils.data import Dataset
from torchvision import transforms


# In[ ]:


class AugmentationDataset(Dataset):
  def __init__(self,df,augmentation):
    self.df=df
    self.augmentations=augmentation
    self.to_tensor=transforms.ToTensor()
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self,idx):
    row=self.df.iloc[idx]
    image_path=row.image_path
    mask_path=row.mask_path
    #print("Image path= ",image_path)
    #print("Mask path= ",mask_path)
    #Load the image and mask
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    mask=cv2.imread(mask_path)
    mask=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
    
    #Apply the augmentations
    if self.augmentations:
      data=self.augmentations(image=image,mask=mask)
      image=data['image']
      mask=data['mask']
    
    #Get the one-hot-encoding
    encoded_mask=get_one_hot_encoding(mask)
    #Convert the image and mask to tensor
    image=self.to_tensor(image)
    encoded_mask=self.to_tensor(encoded_mask)
    
    return image,encoded_mask


# In[ ]:


train_set=AugmentationDataset(df_train,get_augs())
val_set=AugmentationDataset(df_val,get_augs())
train_val_set=torch.utils.data.ConcatDataset([train_set,val_set])
test_set=AugmentationDataset(df_test,get_augs())


# In[ ]:





# In[ ]:


print(f"Train set size={len(train_set)}")
print(f"Val set size={len(val_set)}")
print(f"Train val set size= {len(train_val_set)}")
print(f"Test set size= {len(test_set)}")


# # Custom Dataset Loader

# In[ ]:


from torch.utils.data import DataLoader


# In[ ]:


trainLoader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
valLoader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True)
trainValLoader=DataLoader(train_val_set,batch_size=BATCH_SIZE,shuffle=True)
testLoader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)


# In[ ]:


print(f"# of batches in trainLoader={len(trainLoader)}")
print(f"# of batches in valLoader={len(valLoader)}")
print(f"# of batches in train val loader={len(valLoader)}")
print(f"# of batches in testLoader= {len(testLoader)}")


# # Implement U-Net

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class AttentionLayer(nn.Module):
    def __init__(self, num_channels, reduced_channels):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced_channels, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = self.softmax(y).view(b, -1, 1, 1)
        return x * y
    
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=4, features=[64,128,256,512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attentions_down = nn.ModuleList()  
        self.attentions_up = nn.ModuleList()
        self.attention1=AttentionLayer(features[-1],features[-1])
        self.attention2=AttentionLayer(features[-1]*2,features[-1]*2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.attentions_down.append(AttentionLayer(feature, feature))  
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            self.attentions_up.append(AttentionLayer(feature, feature))  

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 3,1,1,bias=False)

    def forward(self, x):
        skip_connections = []
        
        #print("Down conv part ")
        for idx, (down, attention) in enumerate(zip(self.downs, self.attentions_down)):
            x = down(x)
            x = attention(x)
            skip_connections.append(x)
            x = self.pool(x)
            #print(f"idx= {idx} shape={x.shape}")
        #print("Dim after attention: ",x.shape)
        x = self.attention1(x)
        x = self.bottleneck(x)
        x = self.attention2(x)
        skip_connections = skip_connections[::-1]
        
        #print("Up conv part ")
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            #print(f"idx={idx} shape={x.shape}")

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            x = self.attentions_up[idx // 2](x)  
        return self.final_conv(x)


# In[ ]:


def init_weights_xavier_normal(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.BatchNorm2d):
        init.constant_(layer.weight, 1.0)
        init.constant_(layer.bias, 0.0)


# # Train and Validate

# In[ ]:


model = UNET(in_channels=3, out_channels=4).to(DEVICE)
#Apply weight initialization
model.apply(init_weights_xavier_normal)
loss_func =nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

scaler = torch.cuda.amp.GradScaler()
clip_value=1.6


# In[ ]:


#Choose a part to pass to the U-NET
import random
def choose(L):
    x=random.choice(L)
    L.remove(x)
    return x


# In[ ]:


def trainer(dataloader,model,optimizer,scaler,loss_fn,clip_value):
    model.train()
    total_loss=0
    for images,masks in tqdm(dataloader):
        
        #Split into 4 parts to reduce dimensionality
        images_parts_list,masks_parts_list=split_into_sixteen(images,masks)
        L=list(range(0,16))
        while(len(L)!=0):
            choice=choose(L)
            images_part=images_parts_list[choice]
            masks_part=masks_parts_list[choice]
            
            images_part=images_part.to(DEVICE)
            masks_part=masks_part.to(DEVICE)
            
            # forward            
            with torch.cuda.amp.autocast():
                predictions = model(images_part)
                #print("predictions shape= ",predictions.shape)
                #print("Mask shape= ",masks_part.shape)
                loss = loss_fn(predictions, masks_part)
                #print("Loss= ",loss)
        
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # Clip gradients based on L2 norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
            total_loss+=loss.item()
        #print("batch processed!")
    print(f"Mean Training Loss={total_loss /len(dataloader)}")


# In[ ]:


#Test run 
#trainer(trainLoader,model,optimizer,scaler,loss_func) Working!


# In[ ]:


def convert_to_one_hot(predicted):
    # Find the class with the highest probability along the channel dimension
    max_indices = torch.argmax(predicted, dim=1)

    # Convert the max_indices tensor to one-hot encoding
    one_hot_predicted = F.one_hot(max_indices, num_classes=predicted.size(1))

    # Rearrange dimensions to match the original tensor (n, c, h, w)
    one_hot_predicted = one_hot_predicted.permute(0, 3, 1, 2).to(predicted.dtype)

    return one_hot_predicted


# In[ ]:


def dice_score(predicted, target, eps=1e-7):
    assert predicted.size() == target.size(), "Input tensors must have the same shape"
    
    # Initialize an empty tensor to store the Dice scores for each class
    dice_scores = torch.zeros(predicted.size(1), device=predicted.device, dtype=predicted.dtype)

    # Iterate over each class and calculate the Dice score
    for class_idx in range(predicted.size(1)):
        pred_class = predicted[:, class_idx].contiguous().view(predicted.size(0), -1)
        target_class = target[:, class_idx].contiguous().view(target.size(0), -1)

        intersection = (pred_class * target_class).sum(dim=1)
        volumes = pred_class.sum(dim=1) + target_class.sum(dim=1)

        dice_scores[class_idx] = ((2 * intersection + eps) / (volumes + eps)).mean()

    return dice_scores


# In[ ]:


def check_performance(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 1920*1080*BATCH_SIZE*len(loader)
    total_ds = 0
    rows=4
    cols=4
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(loader):
            #Split each image and masks in batch into 16 parts
            images_parts_list,masks_parts_list=split_into_sixteen(images,masks)
            
            masked_preds_list=[]
            dice_class=0
            #print("Checking accuracy")
            for i in range(16):
                images_part=images_parts_list[i]
                masks_part=masks_parts_list[i]
                
                images_part=images_part.to(DEVICE)
                masks_part=masks_part.to(DEVICE)
                #print(f"image part shape={images_part.shape}")
                preds = model(images_part)
                
                preds = F.softmax(preds, dim=1)
                max_indices = torch.argmax(preds, dim=1)
                masked_preds=convert_to_one_hot(preds)
                #print(f"image part(masked pred) shape={masked_preds.shape}")
                masked_preds_list.append(masked_preds)
                
                arg_masks =torch.argmax(masks_part,dim=1)
                num_correct += (max_indices == arg_masks).sum()
            
            mask_rows=[]
            for i in range(rows):
                mask_row=torch.cat(masked_preds_list[i*cols: (i+1)*cols],dim=-1)
                mask_rows.append(mask_row)
   
            #upper_part_masked_pred=torch.cat((masked_preds_list[0],masked_preds_list[1]),dim=3)
            #print("Upper part shape= ",upper_part_masked_pred.shape)
            #lower_part_masked_pred=torch.cat((masked_preds_list[2],masked_preds_list[3]),dim=3)
            #print("Lower part shape= ",lower_part_masked_pred.shape)
            
            #whole_masked_pred=torch.cat((upper_part_masked_pred,lower_part_masked_pred),dim=2)#Along height dim
            whole_masked_pred=torch.cat(mask_rows,dim=-2)
            #print("Whole predicted mask shape= ",whole_masked_pred.shape)
            #upper_part_masked_true=torch.cat((masks_parts_list[0],masks_parts_list[1]),dim=3)
            #lower_part_masked_true=torch.cat((masks_parts_list[2],masks_parts_list[3]),dim=3)
            whole_masked_true=masks.to(DEVICE)
            #print("Whole true mask shape= ",whole_masked_true.shape)
            ds=sum(dice_score(whole_masked_pred,whole_masked_true))/whole_masked_true.shape[1]
            #print("ds= ",ds)
            total_ds+=ds
            
                
    acc=num_correct/num_pixels
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    mds=total_ds/len(loader)
    print(f"Mean Dice score: {mds}")
    model.train()
    return mds,acc


# In[ ]:





# # Saving and Loading

# In[ ]:


def save_checkpoint(state, filename="checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"],strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch=checkpoint['epoch']
    print(f"start epoch= ",start_epoch)
    return start_epoch


# # Training

# In[ ]:


get_ipython().run_line_magic('cp', '/kaggle/input/unetsmallestcheckpoint/checkpoint.pth .')
get_ipython().run_line_magic('ls', '')


# In[ ]:


mds=0 #
if os.path.exists("checkpoint.pth"):
    print("Exists")
    LOAD_MODEL=True


if LOAD_MODEL:
    start_epoch=load_checkpoint(torch.load("checkpoint.pth"), model)
    print("Model loaded")
else:
    start_epoch=0
#Accuracy of a random model -Testing
mds,acc=check_performance(testLoader, model, device=DEVICE)

#Loop
for epoch in range(start_epoch,EPOCHS):
    print(f"EPOCH #{epoch}")
    print("*"*100)
    #call train
    trainer(trainValLoader,model,optimizer,scaler,loss_func,clip_value)
    #Save model
    checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "epoch":epoch+1,
        }
    save_checkpoint(checkpoint)
    
    #Check accuracy
    re_mds,re_acc=check_performance(testLoader,model,device=DEVICE)
    #Save best model 
    if (re_mds>mds):
        save_checkpoint(checkpoint,filename="best_checkpoint.pth")
        print(f"Best checkpoint found at epoch={epoch}")
        mds=re_mds
    # Update the learning rate using ROPLR
    val_loss=1-re_mds
    #scheduler.step(val_loss)


# # Pretrain model - test

# In[ ]:


"""from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

ENCODER="efficientnet-b2"
WEIGHT="imagenet"
"""


# In[ ]:


"""class PretrainedModel(nn.Module):
    def __init__(self):
        super(PretrainedModel,self).__init__()
        self.architecture=smp.Unet(encoder_name=ENCODER,
                                  encoder_weights=WEIGHT,
                                    in_channels=3,
                                   classes=4,
                                  activation='softmax')
    def forward(self,images,masks=None):
        soft_prob=self.architecture(images)
        if (masks):
            loss=DiceLoss('multiclass')(soft_prob,masks)
            return soft_prob,loss
        return soft_prob
"""

