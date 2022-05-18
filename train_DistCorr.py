"""Train script for DC"""
import argparse
import logging
import os
import sys
import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from evals.eval_DistCorr import eval_net
from dataset.DistortCorrectDataset import DistCorrectDataset


from models.half_unet import UNet_half_4_dc



from utils.utils import log_init
from utils.loss_util import cal_polynomial_model_loss, cal_radial_model_loss
from settings.settings import *

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import conv2d


train_txt_path = r''
dir_checkpoint = r''


def train_net(net,
              device,
              epochs = 5,
              batch_size=4,
              lr=0.001,
              val_percent=0.1,
              l1loss=8,
              l2loss=2,
              loss_mod='l1+l2',
              save_cp=True,
              img_size=128,
              dist_model = 'poly',
              order = 1,
              model = 'UNet',
              para_loss = False):

    # net = net.to(device)
    dataset = DistCorrectDataset(train_txt_path, img_size=img_size)

    n_val = int(len(dataset)*val_percent)
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir=r'',comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Input size:  {img_size}
        Model:  {model}
        Distortion model: {dist_model}
        Polynomial Order: {order}
    ''')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)
    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)



    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=80) as pbar:
            for batch in train_loader:
                imgs = batch['image']
                corner = batch['corner']
                assert imgs.shape[1] == 1,\
                    f'Network input must be in 1 channel, while input channel is {imgs.shape[1]}'

                imgs = imgs.to(device=device, dtype=torch.float32)
                
                corner = corner.to(device=device, dtype=torch.float32)

                net_out = net(imgs)
                parameters = net_out['parameters']
                if model == 'UNet_half':
                    heatmap = imgs
                else:
                    heatmap = net_out['heatmap']

                #=====================================================loss part start
                
                if dist_model == 'poly':
                    loss = cal_polynomial_model_loss(batch_size, order, parameters, corner, device)
                elif dist_model == 'radi':
                    loss = cal_radial_model_loss(batch_size, order, parameters, corner, device)

                
                #KL loss
                # loss = criterion_kl(heatmap_pred, true_heatmap)

                #=====================================================loss part end
                epoch_loss += loss
                writer.add_scalar('Loss/train', loss, global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

                #========================================add to tensorboard
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:

                    val_score, heatmap_pred_val = eval_net(net, val_loader, device, model, batch_size, order, dist_model)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)


                    logging.info('Validation loss: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)


                    writer.add_images('images', imgs, global_step)
    
                    writer.add_images('heatmaps/pred', heatmap, global_step)

                    # break

        if save_cp:
            try:
                os.mkdir(os.path.join(dir_checkpoint,model))
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            temp_path = r''
            torch.save(net.state_dict(),
                       os.path.join(os.path.join(dir_checkpoint,model), f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()



def get_args():
    """Input
    """
    parser = argparse.ArgumentParser(description='Train the distortion correction network on distorted images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=40,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img-size', dest='imgsize', type=int, default=480,
                        help='Size of input image')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-m', '--model', dest='model', type=str, default='UNet_half',
                        help='the model this net takes')
    parser.add_argument('-l1', '--l1loss', dest='l1loss', type=float, default=0.5,
                        help='l1 loss coffience')
    parser.add_argument('-l2', '--l2loss', dest='l2loss', type=float, default=0.5,
                        help='l2 loss coffience')
    parser.add_argument('-lm', '--lossmod', dest='lm', type=str, default='l2',
                        help='loss mode')
    parser.add_argument('-o', '--order', dest='order', type=int, default=3,
                        help='polynomial order')              
    parser.add_argument('-dm', '--dist_model', dest='distmodel', type=str, default='radi',
                        help='distortion model')  

    return parser.parse_args()



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    log_file = os.path.join(LOGFILEPATH , 'trainlog.txt')
    log_init(log_file)

    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    

    args.load = r''

    if args.model == 'UNet_half':
        net = UNet_half_4_dc(args.order, device, 1, 1, True, img_size=args.imgsize, model=args.distmodel)
        args.load = None

    #================================================================================
    #Load parameters part

    

    # separate train-------------------------------------
    # Need Change the Model part
    if not args.load:
        logging.warning(f'No pretrained net wetght!')
    else:
        dict_trained = torch.load(args.load)
        dict_new = net.state_dict().copy()

        new_list = list (net.state_dict().keys())
        pre_trained_list = list (dict_trained.keys())

        for i in range(len(pre_trained_list)):
            dict_new[new_list[i]] = dict_trained[pre_trained_list[i]]
        
        net.load_state_dict(dict_new)

    # # unite train----------------------------------------------------
    # dict_trained = torch.load(args.load)

    # net.load_state_dict(dict_trained)

    #================================================================================

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_size=args.imgsize,
                  val_percent=args.val / 100,
                  l1loss=args.l1loss,
                  l2loss=args.l2loss,
                  loss_mod=args.lm,
                  order=args.order,
                  dist_model=args.distmodel,
                  model=args.model)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)