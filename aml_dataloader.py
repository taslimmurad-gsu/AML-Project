import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from engine import train_one_epoch, evaluate
#import utils
import transforms as T
import os
class RaccoonDataset2(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms=None):
      self.root = root
      self.transforms = transforms
      self.imgs = sorted(os.listdir(os.path.join(root, "images")))
      self.path_to_data_file = data_file
      
    def __getitem__(self, idx):
      # load images and bounding boxes
      img_path = os.path.join(self.root, "images", self.imgs[idx])
      img = Image.open(img_path).convert("RGB")
      box_list = parse_one_annot2(self.path_to_data_file, 
      self.imgs[idx])
      boxes = torch.as_tensor(box_list, dtype=torch.float32)
      num_objs = len(box_list)
      # there is only one class
      #labels = torch.ones((num_objs,), dtype=torch.int64)
      labels= parse_label(self.path_to_data_file, self.imgs[idx])
      
      #labels= torch.from_numpy(data["class_id"].values)

      image_id = torch.tensor([idx])
      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,
      0])
      # suppose all instances are not crowd
      iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
      target = {}
      target["boxes"] = boxes
      target["labels"] = labels
      target["image_id"] = image_id
      target["area"] = area
      target["iscrowd"] = iscrowd
      if self.transforms is not None:
         img, target = self.transforms(img, target)
      return img, target
    def __len__(self):
         return len(self.imgs)


def get_transform2(train):
   transforms = []
   # converts the image, a PIL image, into a PyTorch Tensor
   transforms.append(T.ToTensor())
   if train:
      # during training, randomly flip the training images
      # and ground-truth for data augmentation
      transforms.append(T.RandomHorizontalFlip(0.5))
   return T.Compose(transforms)

def parse_one_annot2(path_to_data_file, filename):
   data = pd.read_csv(path_to_data_file)
   ab= data["image_id"] + '.jpg'
   boxes_array = data[ab == filename][["x_min", "y_min", "x_max", "y_max"]].values
   return boxes_array

def parse_label(path_to_data_file, filename):
   data = pd.read_csv(path_to_data_file)
   ab= data["image_id"] + '.jpg'
   label = data[ab == filename]["class_id"].values
   label = torch.from_numpy(label)
   return label


# use our dataset and defined transformations

dataset2 = RaccoonDataset2(root= "/content/drive/MyDrive/GoogleColab/Datasets/VinBigData", data_file= "/content/drive/MyDrive/GoogleColab/Datasets/VinBigData/train2.csv",transforms = get_transform(train=True))
#dataset2 = RaccoonDataset2(root= "/content/drive/MyDrive/GoogleColab/Code/AML_OBJ_DET/od_dir/raccoon_dataset", data_file= "/content/drive/MyDrive/GoogleColab/Code/AML_OBJ_DET/od_dir/raccoon_dataset/data/raccoon_labels.csv",transforms = get_transform(train=True))

#dataset2.__getitem__(0)[1]['labels']  #print label of image at zero index.
#dataset2.__getitem__(0)[1]
#dataset2.__getitem__(0)[1]['boxes']
#dataset2.__getitem__(0)[0].shape   #image pixels
dataset2.__getitem__(0)[1]['boxes']

# define training and validation data loaders
data_loader2 = torch.utils.data.DataLoader(
              dataset2, batch_size=3, shuffle=True, num_workers=4,
              collate_fn=utils.collate_fn)

len(data_loader2)

