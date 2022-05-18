""" pytorch dataset class """

from PIL import Image
import numpy as np 
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import torchvision.transforms as transforms

class ChessboardDetectDataset(Dataset):
    """Dataset for chessboard corner detector

    func:
        :
    """
    def __init__(self, txt_path, img_size=128):
        """dataset
            txt saved as: img_path heatmap_path
        """
        self.img_paths = []
        self.heatmap_paths = []
        self.norm_size = img_size

        self.txt_path = txt_path
        with open(self.txt_path, 'r') as f:
            paths = f.readlines()

        for i in range(len(paths)):
            path_list = paths[i].split(' ')
            self.img_paths.append(path_list[0])
            self.heatmap_paths.append(path_list[1][:-1])
        
        
        logging.info('Load %d imgs and %d heatmaps from %s.'%(len(self.img_paths), len(self.heatmap_paths), '\\'.join(paths[0].split(' ')[0].split('\\')[:-2])))
    

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
    def preprocess_heatmap(self, heatmap, norm_size):
        """
        """
        # pil_img = Image.fromarray(heatmap)
        # pil_img = pil_img.resize((norm_size, norm_size))
        # heatmap = np.array(pil_img)
        #=====================normalize heatmap
        # everypix /= sum(pix) add 11 12
        # heatmap /= np.sum(heatmap)
        # heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap)-np.min(heatmap))
        

        if len(heatmap.shape) == 2:
            heatmap = np.expand_dims(heatmap, axis=2)
        
        heatmap = heatmap.transpose((2,0,1))

        heatmap = torch.from_numpy(heatmap).type(torch.FloatTensor)

        return heatmap

    def __getitem__(self, index):
        """No Normalization
        """
        img_name = self.img_paths[index]
        heatmap_name = self.heatmap_paths[index]

        img = Image.open(img_name)
        img = img.convert('L')
        heatmap = np.load(heatmap_name)
        # print(np.max(heatmap))


        assert img.size == heatmap.shape, \
            f'Image and heatmap {index} should be the same size, but are {img.size} and {heatmap.size}'

        img = self.preprocess(img, self.norm_size)
        heatmap = self.preprocess_heatmap(heatmap, self.norm_size)
        
        # Normal & ToTensor
        transform_toTensor = transforms.ToTensor()
        img = transform_toTensor(img)



        return { 
            'image' : img,
            'heatmap': heatmap
        }



    def __len__(self):
        return len(self.img_paths)

    