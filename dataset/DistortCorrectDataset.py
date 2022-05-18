from PIL import Image
import numpy as np 
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import torchvision.transforms as transforms

class DistCorrectDataset(Dataset):
    def __init__(self, txt_path, img_size=128):
        """dataset
            txt saved as: img_path corner_path
            Every N corner is 3xNx3, the first one in dim=0 is corner before, the second one in dim=0 is corner after
                corner[2,0,0] = distort parameter
        """
        self.img_paths = []
        self.corner_paths = []
        self.norm_size = img_size

        self.txt_path = txt_path
        with open(self.txt_path, 'r') as f:
            paths = f.readlines()

        for i in range(len(paths)):
            path_list = paths[i].split(' ')
            self.img_paths.append(path_list[0])
            self.corner_paths.append(path_list[1][:-1])
        
        
        logging.info('Load %d imgs and %d corner from %s.'%(len(self.img_paths), len(self.corner_paths), '\\'.join(paths[0].split(' ')[0].split('\\')[:-2])))
    

    @classmethod
    def preprocess(self, pil_img, norm_size):
        """resize & expand dim on 2 
        """
        pil_img = pil_img.resize((norm_size, norm_size))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_nd

        if img_trans.max() > 1:
            img_trans = img_trans / 255
        else:
            img_trans = img_trans / img_trans.max()

        return img_trans
    
    @classmethod
    def filter_the_corners(self, corner):
        """
        Args:
            corner: 2xNx3
        Return:
            corner[diff>1]
        """
        corner_before = corner[0]
        corner_after = corner[1]
        pass




    def __getitem__(self, index):
        """No Normalization
        """
        img_name = self.img_paths[index]
        corner_name = self.corner_paths[index]

        img = Image.open(img_name)
        corners = np.load(corner_name)

        img = self.preprocess(img, self.norm_size)
        
        # Normal & ToTensor
        transform_toTensor = transforms.ToTensor()
        img = transform_toTensor(img)



        return { 
            'image' : img,
            'corner': torch.from_numpy(corners).type(torch.FloatTensor) # 3xNx2 - the first one in dim=0 is corner before, the second one in dim=0 is corner after & the corner[2,0,0]=dist_para
        }



    def __len__(self):
        return len(self.img_paths)