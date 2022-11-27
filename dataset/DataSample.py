"""used to sample data and construct dataset """

import os 
import cv2
import shutil
import logging
import glob
import numpy as np

from utils.utils import log_init, test_dir_if_not_create
from settings.settings import *

class DatasetConstructerDC(object):
    """ For Distortion Correction

        img_list = [img_filefolders]
        corner_list = [corner_filefolders]
        dst = '~\DCDataset'

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                               !Important Information!
        Every N corner is 2xNx3, the first one in dim=0 is corner before, the second one in dim=0 is corner after
        Therefore
        corner filefolders should be [
            [corner_before file list],
            [corner_after file list]
        ]
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        -> DatasetConstructer.sample_data_for_dist_correct(img_list, [corner_before, corner_after], dst)

        -> 
            |-dst path
            |  |-train
            |  | |-img
            |  | | |- ...
            |  | |-corner
            |  | | |- ...
            |  | |-txtfile
            |  |-test
            |    |-...

    Func:
        move files
        spilit data
        write txt


    """
    def __init__(self):
        """
        Args:
            dst file:
                |-dst path
                |  |-train
                |  | |-img
                |  | | |- ...
                |  | |-corner_after
                |  | | |- ...
                |  | |-corner-before
                |  | | |-...
                |  | |-txtfile
                |  |-test
                |    |-...
            src_path:
                [[img_paths],[corner_path],...]
        """
        log_init(os.path.join(LOGFILEPATH,'DataMovelog.txt'))
        self.dst_path = r''
        self.src_path = []
        self.src_img_path = []
        self.src_corner_path = []
        self.ratio = RATIO
        pass

    def sample_data_for_distortion_correction(self, img_file_folders, corner_file_folders, dst_path):
        """
            sample img files and corner files
            1. spilit files
            2. stack corners
            3. move files
            4. write txt
        Args:
            img_file_folders = [...]
            corner_file_folders = [[...], [...]]
        """
        #===================== set
        train_img_dst, train_corner_dst, test_img_dst, test_corner_dst = self.__create_dst_folders_for_corner_detect(dst_path)

        #===================== move imgs
        #============== spilit
        train_img_list, test_img_list = self.__spilit_filefolders(img_file_folders,self.ratio)
        #============== move
        self.__move_files(train_img_list, train_img_dst, 'jpg')
        self.__move_files(test_img_list, test_img_dst, 'jpg')

        # # Now we don't need to stack now 20-11-29
        # train_corner_list, test_corner_list = self.__spilit_filefolders(corner_file_folders, self.ratio)
        # #============= move
        # self.__move_files(train_corner_list, train_corner_dst, 'npy')
        # self.__move_files(test_corner_list, test_corner_dst, 'npy')

        #==================== read and stack corners
        #============== spilit
        train_corner_before_list, test_corner_before_list = self.__spilit_filefolders(corner_file_folders[0], self.ratio)
        train_corner_after_list, test_corner_after_list = self.__spilit_filefolders(corner_file_folders[1], self.ratio)

        self.stack_and_save_corners(train_corner_before_list, train_corner_after_list, train_corner_dst)
        self.stack_and_save_corners(test_corner_before_list, test_corner_after_list, test_corner_dst)

        #=================== write
        train_file_path = os.path.join(dst_path, 'train')
        test_file_path = os.path.join(dst_path, 'test')
        self.__write_txt_for_corner_detect(train_file_path, 'train')
        self.__write_txt_for_corner_detect(test_file_path, 'test')
  
    def stack_and_save_corners(self, corner_before_list, corner_after_list, dst_path):
        """
            corner should be saved as Nx3 np.ndarray
        """
        assert len(corner_before_list) == len(corner_after_list)
        file_num = len(glob.glob(os.path.join(dst_path, '*.npy')))
        logging.info(f'Save corner from {file_num} ...' )

        
        for i, corner_before in enumerate(corner_before_list):
            corner_before =  np.load(corner_before)
            corner_after = np.load(corner_after_list[i])

            assert corner_before.shape == corner_after.shape

            assert corner_after.shape[1] == 3

            res_temp = np.zeros((2,corner_after.shape[0],3))
            res_temp[0] = corner_before
            res_temp[1] = corner_after
            
            
            np.save(os.path.join(dst_path, str(i+file_num)+'.npy'), res_temp)

    def write_txt(self,folder_path):
        """
        """
        self.__write_txt_for_corner_detect(folder_path)
        
    def __write_txt_for_corner_detect(self,folder_path, task='train'):
        """
            |-train/test
                |-img
                |-corner
        """
        filename = os.path.join(folder_path, task+'.txt')

        imgs_path = os.path.join(folder_path, 'img')
        corners_path = os.path.join(folder_path, 'corner')

        img_name_list = os.listdir(imgs_path)
        corners_list = os.listdir(corners_path)

        assert len(img_name_list) == len(corners_list), f'path is :{folder_path}'

        with open(filename, 'w') as f:
            for i, img_name in enumerate(img_name_list):
                f.write(os.path.join(imgs_path,img_name)+' '+os.path.join(corners_path,corners_list[i])+'\n')

    def __create_dst_folders_for_corner_detect(self, dst):
        """
        """
        
        #-train
        train_file_path = os.path.join(dst,'train')
        test_dir_if_not_create(train_file_path)
        #--img
        train_img_path = os.path.join(train_file_path,'img')
        test_dir_if_not_create(train_img_path)

        #--corner
        train_corner_path = os.path.join(train_file_path,'corner')
        test_dir_if_not_create(train_corner_path)
        
        # test
        test_file_path = os.path.join(dst,'test')
        test_dir_if_not_create(test_file_path)
        #--img
        test_img_path = os.path.join(test_file_path,'img')
        test_dir_if_not_create(test_img_path)

        #--corner   
        test_corner_path = os.path.join(test_file_path,'corner')
        test_dir_if_not_create(test_corner_path)
        
        return train_img_path, train_corner_path, test_img_path, test_corner_path

    # COMMONLY USED
    def __spilit_filefolders(self, file_folders, ratio):
        """spilit files into train, val, test 
            at ratio
            spilit train and val can be performed by torch.utils.data.random_split
                random_split(full_data, [train_size, val_size])
        Args:
            file_folders: file folders absolute paths
            ratio: [train, test]
        Returns:
            train_file_list: train files in ABSOLUTE path
            test_file_list: test files in ABSOLUTE path
        """
        train_file_list = []
        test_file_list = []

        for file_folder in file_folders:
            train_file_list_temp, test_file_list_temp = self.__spilit_filefolder(file_folder, ratio)
            
            train_file_list.extend(train_file_list_temp)
            test_file_list.extend(test_file_list_temp)
        
        logging.info("Totally spilit files into %d for train and %d for test" % (len(train_file_list), len(test_file_list)))

        return train_file_list, test_file_list     

    # COMMONLY USED
    def __spilit_filefolder(self, file_folder, ratio):
        """spilit single filefloder
        Args:
            file_folder: file folder ABSOLUTE path
            ratio: [train, test]

        Returns:
            train_file_list: train files in ABSOLUTE path
            test_file_list:
        """
        file_name_list = os.listdir(file_folder)
        file_num = len(file_name_list)
        # print(file_num)
        ratio_sum = ratio[0]+ratio[1]
        train_num = file_num // ratio_sum * ratio[0]
        test_num = file_num // ratio_sum * ratio[1]
        # print(train_num)

        train_file_list = [os.path.join(file_folder,i) for i in file_name_list[:train_num]]
        test_file_list = [os.path.join(file_folder,i) for i in file_name_list[train_num:]]

        return train_file_list, test_file_list

    # COMMONLY USED
    def __move_files(self, src_list, dst, file_postfix = 'jpg'):
        """
        Args:
            src: a list of files' absolute path
            dst: a folder
                 rule of name: i.xxx
        """
        dst_file_list = os.listdir(dst)
        cur_len = len(dst_file_list)

        src_file_num = len(src_list)

        logging.info('Move %d %s files from %s to %s saved from %d.file' % (src_file_num, file_postfix, src_list[0].split('\\')[:-1], dst, cur_len))

        
        for i, srcfile in enumerate(src_list):
            if srcfile.split('.')[-1]!= file_postfix:
                continue
            else:
                dst_file_name = os.path.join(dst, str(i+cur_len)+'.'+file_postfix)
                shutil.copyfile(srcfile, dst_file_name)

class DatasetConstructerCD(object):
    """ for Chessboard Corner Detection 
        img_list = [img_filefolders]
        heatmap_list = [heatmap_filefolders]
        dst = r'~\CornerDetectDataset'

        -> DatasetConstructer.sample_data_for_corner_detect(img_list, heatmap_list, dst)

        -> 
            |-dst path
            |  |-train
            |  | |-img
            |  | | |- ...
            |  | |-heatmap
            |  | | |- ...
            |  | |-txtfile
            |  |-test
            |    |-...

    Func:
        move files
        spilit data
        write txt


    """
    def __init__(self):
        """
        Args:
            dst file:
                |-dst path
                |  |-train
                |  | |-img
                |  | | |- ...
                |  | |-heatmap
                |  | | |- ...
                |  | |-txtfile
                |  |-val
                |  |-test
            src_path:
                [[img_paths],[heatmap_paths],[corner_path],...]
        """
        log_init(os.path.join(LOGFILEPATH,'DataMovelog.txt'))
        self.dst_path = r''
        self.src_path = []
        self.src_img_path = []
        self.src_heatmap_path = []
        self.ratio = RATIO
        pass

    def sample_data_for_corner_detect(self, img_file_folders, heatmap_file_folders, dst_path):
        """
            sample img files and heatmap files
            1. spilit files
            2. move files
            3. write txt
        """
        #===================== set
        train_img_dst, train_heatmap_dst, test_img_dst, test_heatmap_dst = self.__create_dst_folders_for_corner_detect(dst_path)

        #===================== move imgs
        #============== spilit
        train_img_list, test_img_list = self.__spilit_filefolders(img_file_folders,self.ratio)
        #============== move
        self.__move_files(train_img_list, train_img_dst, 'jpg')
        self.__move_files(test_img_list, test_img_dst, 'jpg')

        #==================== move heatmaps
        #============== spilit
        train_heatmap_list, test_heatmap_list = self.__spilit_filefolders(heatmap_file_folders, self.ratio)
        #============== move
        self.__move_files(train_heatmap_list, train_heatmap_dst, 'npy')
        self.__move_files(test_heatmap_list, test_heatmap_dst, 'npy')

        #=================== write
        train_file_path = os.path.join(dst_path, 'train')
        test_file_path = os.path.join(dst_path, 'test')
        self.__write_txt_for_corner_detect(train_file_path, 'train')
        self.__write_txt_for_corner_detect(test_file_path, 'test')
  
    def write_txt(self,folder_path):
        """
        """
        self.__write_txt_for_corner_detect(folder_path)
        
    def __write_txt_for_corner_detect(self,folder_path, task='train'):
        """
            |-train/test
                |-img
                |-heatmap
        """
        filename = os.path.join(folder_path, task+'.txt')

        imgs_path = os.path.join(folder_path, 'img')
        heatmaps_path = os.path.join(folder_path, 'heatmap')

        img_name_list = os.listdir(imgs_path)
        heatmaps_list = os.listdir(heatmaps_path)

        assert len(img_name_list) == len(heatmaps_list)

        with open(filename, 'w') as f:
            for i, img_name in enumerate(img_name_list):
                f.write(os.path.join(imgs_path,img_name)+' '+os.path.join(heatmaps_path,heatmaps_list[i])+'\n')

    def __create_dst_folders_for_corner_detect(self, dst):
        """
        """
        
        #-train
        train_file_path = os.path.join(dst,'train')
        test_dir_if_not_create(train_file_path)
        #--img
        train_img_path = os.path.join(train_file_path,'img')
        test_dir_if_not_create(train_img_path)

        #--heamap
        train_heatmap_path = os.path.join(train_file_path,'heatmap')
        test_dir_if_not_create(train_heatmap_path)
        
        # test
        test_file_path = os.path.join(dst,'test')
        test_dir_if_not_create(test_file_path)
        #--img
        test_img_path = os.path.join(test_file_path,'img')
        test_dir_if_not_create(test_img_path)

        #--heamap
        test_heatmap_path = os.path.join(test_file_path,'heatmap')
        test_dir_if_not_create(test_heatmap_path)
        
        return train_img_path, train_heatmap_path, test_img_path, test_heatmap_path

    def __spilit_filefolders(self, file_folders, ratio):
        """spilit files into train, val, test
            at ratio
            spilit train and val can be performed by torch.utils.data.random_split
                random_split(full_data, [train_size, val_size])
        Args:
            file_folders: file folders absolute paths
            ratio: [train, test]
        Returns:
            train_file_list: train files in ABSOLUTE path
            test_file_list:
        """
        train_file_list = []
        test_file_list = []

        for file_folder in file_folders:
            train_file_list_temp, test_file_list_temp = self.__spilit_filefolder(file_folder, ratio)
            
            train_file_list.extend(train_file_list_temp)
            test_file_list.extend(test_file_list_temp)
        
        logging.info("Totally spilit files into %d for train and %d for test" % (len(train_file_list), len(test_file_list)))

        return train_file_list, test_file_list        

    def __spilit_filefolder(self, file_folder, ratio):
        """spilit single filefloder
        Args:
            file_folder: file folder ABSOLUTE path
            ratio: [train, test]

        Returns:
            train_file_list: train files in ABSOLUTE path
            test_file_list:
        """
        file_name_list = os.listdir(file_folder)
        file_num = len(file_name_list)
        # print(file_num)
        ratio_sum = ratio[0]+ratio[1]
        train_num = file_num // ratio_sum * ratio[0]
        test_num = file_num // ratio_sum * ratio[1]
        # print(train_num)

        train_file_list = [os.path.join(file_folder,i) for i in file_name_list[:train_num]]
        test_file_list = [os.path.join(file_folder,i) for i in file_name_list[train_num:]]

        return train_file_list, test_file_list

    def __move_files(self, src_list, dst, file_postfix = 'jpg'):
        """
        Args:
            src: a list of files' absolute path
            dst: a folder
                 rule of name: i.xxx
        """
        dst_file_list = os.listdir(dst)
        cur_len = len(dst_file_list)

        src_file_num = len(src_list)

        logging.info('Move %d %s files from %s to %s saved from %d.file' % (src_file_num, file_postfix, src_list[0].split('\\')[:-1], dst, cur_len))

        
        for i, srcfile in enumerate(src_list):
            if srcfile.split('.')[-1]!= file_postfix:
                continue
            else:
                dst_file_name = os.path.join(dst, str(i+cur_len)+'.'+file_postfix)
                shutil.copyfile(srcfile, dst_file_name)

