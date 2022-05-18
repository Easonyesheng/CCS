'''Perform Calibration with accurate sub-pixel points'''

import os
import glob
import numpy as np 
import cv2
import logging
from PIL import Image
from utils.utils import *
from settings import *

class CalibGivenPoints(object):
    """
    """
    def __init__(self, chessboard_size, img_points_path, img_number):
        """
        Args:
            chessboard_size: [,]
            img_points_path: in the folder, each .npy = a set of corner coordinates 
        """
        self.chess_board_size = chessboard_size
        self.chessboard_size = chessboard_size       
        self.criteria = 0
        self.img_points_path = img_points_path
        self.img_number = img_number

#==================================================image calib part
    def load_imglist(self, imglist):
        """given images name list than load images 
        Args:
            imglist - list of absolute path of image            
        """
        assert len(imglist) > 2, f'Images not enough!'
        self.gray_imgs = []
        for img in imglist:
            img_temp = cv2.imread(img, 0)
            img_temp.astype(np.int8)
            self.gray_imgs.append(img_temp)
        print("Load ", len(self.gray_imgs), " images.")

    def __find_corners(self, gray_img):
        """name
            description
        Args:

        Returns:
            ret: whether we find all the corners
        """
        assert len(self.gray_imgs) > 2, f'Images not enough!'
        ret, corners = cv2.findChessboardCorners(gray_img, (self.chess_board_size[1],self.chess_board_size[0]),None)
        return ret, corners

    def __find_corners_subpix(self, img, corners, save_flag=False, name='0'):
        """name
            description
        Args:

        Returns:
        """
        corners2 = cv2.cornerSubPix(img,corners,(11,11), (-1,-1), self.criteria)
        # np.save(os.path.join(self.save_path, name+'npy'), corners2)
        # print(corners2[0])
        return corners2

    def get_points_by_images(self):
        """
        """
        objp_temp = self.get_object_point()

        self.object_points = []
        self.img_points = []

        criteria = self.__pre_set()
        for img in self.gray_imgs:
            ret, corners_temp = self.__find_corners(img)

            if ret:
                self.object_points.append(objp_temp)
                corners2 = self.__find_corners_subpix(img, corners_temp)
                self.img_points.append(corners2)

        

#====================================================    
    def load_img_points(self):
        """load .npy
        Returns:
            img_points: [nparray[corner_numberx1x2],...] - float
            object_points: [nparray[corner_numberx2],...]
        """

        objp_temp = self.get_object_point()

        self.object_points = []
        self.img_points = []
        file_list = glob.glob(os.path.join(self.img_points_path, '*.npy'))
        assert len(file_list) == self.img_number

        logging.info(f'Loading points from {self.img_points_path}')

        for i, img_point_name in enumerate(file_list):
            img_point = np.load(img_point_name)
            assert img_point.shape[0] == objp_temp.shape[0]
            self.img_points.append(img_point.astype('float32'))
            self.object_points.append(objp_temp)
        
        return self.img_points, self.object_points

    def get_object_point(self):
        """name
            get object points
        Args:

        Returns:

        """
        objp = np.zeros((self.chessboard_size[0]*self.chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboard_size[1],0:self.chessboard_size[0]].T.reshape(-1,2)
        return objp
    
    def get_o_points(self):
        """
        """
        objp_temp = self.get_object_point()
        self.object_points = []

        for i in range(self.img_number):
            self.object_points.append(objp_temp)


    def __pre_set(self):
        """name
            termination criteria
        Args:
            

        Returns:

        """
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        return self.criteria
    
    def calibrate(self):
        """name
        Args:

        Returns:
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.img_points, (480, 480),None,None)
        # ret = cv2.initCameraMatrix2D(self.object_points, self.img_points, (480, 480),0)
        return ret, mtx, dist, rvecs, tvecs
    
    def calib_with_fix_IM(self,fx,fy,px,py):
        """name
            cv.calibrateCamera(	objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]
        Args:
        
        Returns:

        """
        cameraMatrix = np.array([[fx,0,px], 
                                [0,fy,py], 
                                [0,0,1]])
        flag = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.img_points, (480, 480),cameraMatrix, None, flags= flag)
        return ret, mtx, dist, rvecs, tvecs

    def call_re_projection_errors(self, rvecs, tvecs, mtx, dist):
        """Respectively
        """
        RPEs = []

        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(self.object_points[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints2 = imgpoints2.squeeze()   
            # print(imgpoints2.shape)
            # print(self.img_points[i].shape)
            error = cv2.norm(self.img_points[i].squeeze(), imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            RPEs.append(error)
        return RPEs

    def run(self):
        """Work flow
        """
        log_init(CALIBLOGFILE)
        logging.info('Starting Calibration...')
        logging.info(f'Chessboard size is {self.chessboard_size}')
    
        self.criteria = self.__pre_set()

        self.load_img_points()

        ret, mtx, dist, rvecs, tvecs = self.calibrate()

        logging.info(f'The reprojection error is: {ret}')
        logging.info(f'intrinsice matrix is: {mtx}')


def load_corners(info_path, corners_path):
    """load info - read points - save points
    """
    info_list = glob.glob(os.path.join(info_path, '*.npy'))

    for info in info_list:
        name = info.split('\\')[-1]
        info = np.load(info, allow_pickle=True)
        info = info.item()
        img_points = []
        img_points_raw = info['img_points']
        for corner in img_points_raw:
            img_points.append([corner[0,0], corner[1,0]])
    
        img_points = np.array(img_points)
        img_points = np.expand_dims(img_points, axis=1)
        print(img_points.shape)
        np.save(os.path.join(corners_path, name), img_points)



if __name__ == '__main__':
    
    # test with ground truth points
    info_path = r'D:\DeepCalib\CalibrationNet\Dataset\For_Traditional_Calib_40_0\info'

    corners_path = r'D:\DeepCalib\CalibrationNet\Dataset\For_Traditional_Calib_40_0\corners'

    # load_corners(info_path, corners_path)

    test = CalibGivenPoints([6,5], corners_path, 40)

    test.run()