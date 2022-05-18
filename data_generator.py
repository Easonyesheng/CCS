'''Data generation script '''
import numpy as np
from numpy import random
import cv2

from dataset.DataGenerator import Checkboard
from utils.utils import *
from settings.settings import *


#=================config template
dataset_gene_config = {
    'pathname': 'template',
    'pose numbers' : 2,
    'fx': [120, 180], # 120~180 
    'fy': [120, 180],
    'px': [150, 180],
    'py': [150, 180],
    'checkboard size' : [(5,5), (5,6), (6,6), (7,8)],
    'd': [3,6],
    'theta': [-35, 35],
    'phi': [-35, 35],
    'x': [-50, 50],
    'y': [-50, 50],
    'z': [0, 120],
    'max item': 400
}

#================================ generate function
def generation_script(config):
    """
    Workflow
        1. determine the save path accroding to the name
        2. generating the data until enough
            2.1 determine parameters randomly in range
            2.2 generating in group (while)
                named as: 'group number'-'pose number'
            2.3 count if inside image
    """
    # STEP 1
    data_save_path = os.path.join(DATASETPATH, config['pathname'])
    test_dir_if_not_create(data_save_path)

    img_save_path = os.path.join(data_save_path, 'img') 
    test_dir_if_not_create(img_save_path)

    info_save_path = os.path.join(data_save_path, 'info')
    test_dir_if_not_create(info_save_path)

    checkboard_save_path = [img_save_path, info_save_path]

    #STEP 2
    data_count = config['start']
    pose_count = 0
    while data_count < config['max']:
        # step 2.1
        #========================== K and checkboard size
        corner_size = config['checkboard size'][0]#[random.randint(3)]

        fx = random.rand()*(config['fx'][1] - config['fx'][0]) + config['fx'][0]
        # fy = random.rand()*(config['fy'][1] - config['fy'][0]) + config['fy'][0]
        fy = fx
        px = random.rand()*(config['px'][1] - config['px'][0]) + config['px'][0]
        # py = random.rand()*(config['py'][1] - config['py'][0]) + config['py'][0]
        py = px

        camera_parameters = {
            'fx': fx,
            'fy': fy,
            'px': px,
            'py': py
        }

        #=========================== pose change
        pose_count = 0
        while pose_count < config['pose numbers']:
            temp_name = str(data_count)+'-'+str(pose_count)
            cur = Checkboard(checkboard_save_path, temp_name, corner_size)

            d = random.rand()*(config['d'][1] - config['d'][0]) + config['d'][0]
            phi = random.rand()*(config['phi'][1] - config['phi'][0]) + config['phi'][0]
            theta = random.rand()*(config['theta'][1] - config['theta'][0]) + config['theta'][0]

            move_parameters = {
                'd': d,
                'theta': theta,
                'phi': phi
            }

            x = random.rand()*(config['x'][1] - config['x'][0]) + config['x'][0]
            y = random.rand()*(config['y'][1] - config['y'][0]) + config['y'][0]
            z = random.rand()*(config['z'][1] - config['z'][0]) + config['z'][0]

            rotate_parameters = {
                'x': x,
                'y': y,
                'z': z
            }

            flag = cur.run(camera_parameters, move_parameters, rotate_parameters, save_flag=True)



            if flag:
                pose_count += 1
            else:
                print('Re generating...')
        
        data_count += 1

def generate_with_pic(config):
    """Generate dataset with images
    for detection
    Workflow:
        1. In certain K, draw a fix checkboard picture => get img_fix & tf
        2. get randomly pose_num x (Rms, tms), draw the move checkboard pic
        3. save 
    """
    data_save_path = os.path.join(DATASETPATH, config['pathname'])
    test_dir_if_not_create(data_save_path)

    img_save_path = os.path.join(data_save_path, 'img') 
    test_dir_if_not_create(img_save_path)

    info_save_path = os.path.join(data_save_path, 'info')
    test_dir_if_not_create(info_save_path)

    checkboard_save_path = [img_save_path, info_save_path]

    data_count = config['start']
    pose_count = 0
    while data_count < config['max']:
        # step 1
        #========================== K and checkboard size
        corner_size = config['checkboard size'][0]#[random.randint(6)]

        fx = random.rand()*(config['fx'][1] - config['fx'][0]) + config['fx'][0]
        # fy = random.rand()*(config['fy'][1] - config['fy'][0]) + config['fy'][0]
        fy = fx
        px = random.rand()*(config['px'][1] - config['px'][0]) + config['px'][0]
        # py = random.rand()*(config['py'][1] - config['py'][0]) + config['py'][0]
        py = px

        camera_parameters = {
            'fx': fx,
            'fy': fy,
            'px': px,
            'py': py
        }

        fix = Checkboard(name='fix', save_path=['',''], corner_size = corner_size)
        fix.camera_load(camera_parameters['fx'],camera_parameters['fy'],camera_parameters['px'],camera_parameters['py'])
        img_fix, tf = fix.draw_fix_checkboard()
        # print(corner_size)
        # img_fix.show()
        # return

        pose_count = 0
        while pose_count < config['pose numbers']:
            temp_name = str(data_count)+'-'+str(pose_count)
            cur = Checkboard(checkboard_save_path, temp_name, corner_size)
            cur.camera_load(camera_parameters['fx'],camera_parameters['fy'],camera_parameters['px'],camera_parameters['py'])

            d = random.rand()*(config['d'][1] - config['d'][0]) + config['d'][0]
            phi = random.rand()*(config['phi'][1] - config['phi'][0]) + config['phi'][0]
            theta = random.rand()*(config['theta'][1] - config['theta'][0]) + config['theta'][0]

            x = random.rand()*(config['x'][1] - config['x'][0]) + config['x'][0]
            y = random.rand()*(config['y'][1] - config['y'][0]) + config['y'][0]
            z = random.rand()*(config['z'][1] - config['z'][0]) + config['z'][0]

            config_move = {
                            'x': x,
                            'y': y,
                            'z': z,
                            'd': d,
                            'phi': phi,
                            'theta': theta,
                            'tf': tf
                        }
            img_bg = None
            flag = cur.draw_move_checkboard(img_fix, img_bg, config_move, save_flag=True)



            if flag:
                pose_count += 1
            else:
                print('Re generating...')
        
        data_count += 1

def generation_script_for_calib(config, fusion_flag=False, dist_flag=False, uneven_light_flag=False):
    """Wraped for Calibration
    Workflow:
        1. In certain K, draw a fix checkboard picture => get img_fix & tf
        2. get randomly pose_num x (Rms, tms), draw the move checkboard pic
        3. save 
    """
    data_save_path = os.path.join(DATASETPATH, config['pathname'])
    test_dir_if_not_create(data_save_path)

    img_save_path = os.path.join(data_save_path, 'img') 
    test_dir_if_not_create(img_save_path)

    info_save_path = os.path.join(data_save_path, 'info')
    test_dir_if_not_create(info_save_path)

    heatmap_save_path = os.path.join(data_save_path, 'heatmap')
    test_dir_if_not_create(heatmap_save_path)

    if dist_flag:
        dist_save_path = os.path.join(data_save_path, 'dist_img')
        test_dir_if_not_create(dist_save_path)
        ori_corner_save_path = os.path.join(data_save_path, 'ori_corner')
        test_dir_if_not_create(ori_corner_save_path)
        dist_corner_save_path = os.path.join(data_save_path, 'dist_corner')
        test_dir_if_not_create(dist_corner_save_path)
        checkboard_save_path = [img_save_path, info_save_path, heatmap_save_path, dist_save_path, ori_corner_save_path, dist_corner_save_path]
    else:
        checkboard_save_path = [img_save_path, info_save_path, heatmap_save_path]

    data_count = config['start']
    pose_count = 0
    while data_count < config['max']:
        # step 1
        #========================== K and checkboard size
        corner_size = config['checkboard size'][0]#[random.randint(6)]

        fx = random.rand()*(config['fx'][1] - config['fx'][0]) + config['fx'][0]
        # fy = random.rand()*(config['fy'][1] - config['fy'][0]) + config['fy'][0]
        fy = fx
        px = random.rand()*(config['px'][1] - config['px'][0]) + config['px'][0]
        # py = random.rand()*(config['py'][1] - config['py'][0]) + config['py'][0]
        py = px

        camera_parameters = {
            'fx': fx,
            'fy': fy,
            'px': px,
            'py': py
        }

        fix = Checkboard(name='fix', save_path=['',''], corner_size = corner_size)
        fix.camera_load(camera_parameters['fx'],camera_parameters['fy'],camera_parameters['px'],camera_parameters['py'])
        img_fix, tf = fix.draw_fix_checkboard(fusion_flag=fusion_flag)
        # print(corner_size)
        # img_fix.show()
        # return

        pose_count = 0
        while pose_count < config['pose numbers']:
            temp_name = str(data_count)+'-'+str(pose_count)
            cur = Checkboard(checkboard_save_path, temp_name, corner_size)
            cur.camera_load(camera_parameters['fx'],camera_parameters['fy'],camera_parameters['px'],camera_parameters['py'])

            d = random.rand()*(config['d'][1] - config['d'][0]) + config['d'][0]
            phi = random.rand()*(config['phi'][1] - config['phi'][0]) + config['phi'][0]
            theta = random.rand()*(config['theta'][1] - config['theta'][0]) + config['theta'][0]

            x = random.rand()*(config['x'][1] - config['x'][0]) + config['x'][0]
            y = random.rand()*(config['y'][1] - config['y'][0]) + config['y'][0]
            z = random.rand()*(config['z'][1] - config['z'][0]) + config['z'][0]

            config_move = {
                            'x': x,
                            'y': y,
                            'z': z,
                            'd': d,
                            'phi': phi,
                            'theta': theta,
                            'tf': tf
                        }
            img_bg = None
            flag = cur.draw_move_checkboard(img_fix, img_bg, config_move, save_flag=True, heatmap_flag=True, fusion_flag=fusion_flag, dist_flag=dist_flag, dist_k=config['dist_k'], uneven_light_flag=uneven_light_flag)



            if flag:
                pose_count += 1
            else:
                print('Re generating...')
        
        data_count += 1


if __name__ == '__main__':
    
    # Example
    config = {
        'pathname': 'test',
        'pose numbers' : 40,
        'dist_k' : [1, -0.45, 0.2],
        'fx': [150, 200], # 120~180 
        'fy': [150, 200],
        'px': [150, 200],
        'py': [150, 200],
        'checkboard size' : [(6,5)],
        'img size' : 480,
        'd': [5,8],
        'theta': [-10,10],
        'phi': [-10, 10],
        'x': [-10, 10],
        'y': [-10, 10],
        'z': [-10, 10],
        'max': 35,
        'start': 3

    }

    generation_script_for_calib(config, fusion_flag=True, dist_flag=True)