import cv2
import numpy as np
import os
import logging



img_name = r''

class OpenCVSubPixelCorner(object):
    """
    """

    def __init__(self, save_path, show_flag=False):
        """
        """
        self.save_path = save_path
        self.show_flag = show_flag
    
    def load_img(self, img_name, corner_num, chessboard_size=[6,5]):
        """
        """
        self.img = cv2.imread(img_name, 0)
        self.name = img_name.split('\\')[-1].split('.')[0]
        self.corner_num = corner_num
        self.chessboard_size = chessboard_size
    
    def __sort_points_list(self, points_list, corner_size):
        """
        """
        points_list = sorted(points_list, key=lambda x:x[0]) # x line horizon

        h, v = corner_size[0], corner_size[1]

        for i in range(v):
            points_list[i*h:(i+1)*h] = sorted(points_list[i*h:(i+1)*h], key=lambda x:x[1])
        
        return points_list
    
    def find_corners(self):
        """
        """
        ret, corners = cv2.findChessboardCorners(self.img, (self.chessboard_size[1],self.chessboard_size[0]),None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(self.img,corners,(11,11), (-1,-1),criteria)
        else:
            logging.info(f'Reject {self.name} Image.')
            return False
        
        self.corner = []
        for i in range(self.chessboard_size[0]*self.chessboard_size[1]):
            self.corner.append([corners2[i,0,1],corners2[i,0,0]])
    
        self.corner = self.__sort_points_list(self.corner, self.chessboard_size)
        self.corner = np.array(self.corner)

        return True

    def find_corners_save(self, name):
        """
        """
        ret, corners = cv2.findChessboardCorners(self.img, (self.chessboard_size[1],self.chessboard_size[0]),None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(self.img,corners,(11,11), (-1,-1),criteria)
            np.save(os.path.join(self.save_path, name+'.npy'), corners2)
        else:
            logging.info(f'Reject {self.name} Image.')
            return False
        

        
        return True

    def load_gt(self, gt_name):
        assert gt_name.split('\\')[-1].split('.')[0] == self.name
        info = np.load(gt_name, allow_pickle=True)
        info = info.item()
        gt = []
        img_points_raw = info['img_points']
        for corner in img_points_raw:
            gt.append([corner[0,0], corner[1,0]])

        self.gt = self.__sort_points_list(gt, self.chessboard_size)
        self.gt = np.array(self.gt)
        return self.gt

    def cal_err(self):
        self.err = 0
        for i in range(self.corner_num):
            # err_t = np.sqrt(np.sum(np.power(self.corner[i]-self.gt[i],2),axis=0))
            err_t = np.sum(np.sum(np.abs(self.corner[i]-self.gt[i]),axis=0)) / self.corner[i].shape[0]
            self.err += err_t

        self.err /= self.corner_num
        return self.err 