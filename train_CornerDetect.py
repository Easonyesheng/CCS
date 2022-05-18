"""Train script for SuperpointNet"""
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

from evals.eval_CornerDetect import eval_net
from dataset.ChessboardData import ChessboardDetectDataset

from models.unet_model import UNet

from utils.utils import log_init
from settings.settings import *

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import conv2d


train_txt_path = r''
# train_txt_path = r''
dir_checkpoint = r''


def train_net(net,
              device,
              epochs = 5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_size=256,
              l1loss=0.5,
              l2loss=0.5,
              loss_mod='l2',
              model = 'UNet'):

    # net = net.to(device)
    dataset = ChessboardDetectDataset(train_txt_path, img_size=img_size)

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
        loss mod: {loss_mod}
        l1weight: {l1loss}
        l2weight: {l2loss}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # loss
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_kl = nn.KLDivLoss(reduction='mean')

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=80) as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_heatmap = batch['heatmap']
                assert imgs.shape[1] == 1,\
                    f'Network input must be in 1 channel, while input channel is {imgs.shape[1]}'

                imgs = imgs.to(device=device, dtype=torch.float32)
                
                heatmap_type = torch.float32
                true_heatmap = true_heatmap.to(device=device, dtype=heatmap_type)

                if model=='Superpoint':
                    net_out = net(imgs)
                    params = {
                        'out_num_points': 500,
                        'patch_size': 1,
                        'device': device,
                        'nms_dist': 4,
                        'conf_thresh': 0.015
                    }

                    sp_processer = SuperPointNet_process(**params)
                    outs = net.process_output(sp_processer)
                    heatmap_pred = outs['heatmap']
                elif model == 'UNet'or model == 'UNetSimp':
                    heatmap_pred = net(imgs)

                #=====================================================loss part start
                # heatmap_pred_patch_loss = patch_loss(heatmap_pred, device=device ,batch= batch_size).to(device=device, dtype=heatmap_type)
                # true_heatmap_patch_loss = patch_loss(true_heatmap, device=device ,batch=batch_size).to(device=device, dtype=heatmap_type)
                # l2
                # loss = criterion_mse(heatmap_pred, true_heatmap)

                #l1+l2
                loss = l2loss*criterion_mse(heatmap_pred, true_heatmap) + l1loss*criterion_l1(heatmap_pred, true_heatmap)#torch.sum(torch.sum(torch.abs(heatmap_pred-true_heatmap)))

                #KL loss
                # loss = criterion_kl(heatmap_pred, true_heatmap)

                #=====================================================loss part end
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

                #========================================add to tensorboard
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    # for tag, value in net.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score, heatmap_pred_val = eval_net(net, val_loader, device, model=model)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)


                    logging.info('Validation MSE loss: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)


                    writer.add_images('images', imgs, global_step)
    
                    writer.add_images('heatmaps/true', true_heatmap, global_step)
                    writer.add_images('heatmaps/pred', heatmap_pred, global_step)

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

def patch_loss(data, batch, device, patch_size=3):
    """calculate the patch loss

    Args:
        data: tensor [batch, channel=1, W, H] if batch else [channel=1, W, H]
    """
    
    kernel =[[0.03797616, 0.044863533, 0.03797616],
         [0.044863533, 0.053, 0.044863533],
         [0.03797616, 0.044863533, 0.03797616]]


    if batch>1:
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(1,1,3,3)
        weight = nn.Parameter(data=kernel, requires_grad=False)
        weight = weight.to(device)

        data_res = conv2d(data, weight, stride=1, padding=1, dilation=1)
        # print(data_res.shape)

    else:
        pass 


    return data_res


def get_args():
    """Input
    """
    parser = argparse.ArgumentParser(description='Train the detection network on images and target heatmaps',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=r'',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img-size', dest='imgsize', type=int, default=480,
                        help='Size of input image')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-m', '--model', dest='model', type=str, default='UNet',
                        help='the model this net takes')
    parser.add_argument('-l1', '--l1loss', dest='l1loss', type=float, default=0.5,
                        help='l1 loss coffience')
    parser.add_argument('-l2', '--l2loss', dest='l2loss', type=float, default=0.5,
                        help='l2 loss coffience')
    parser.add_argument('-lm', '--lossmod', dest='lm', type=str, default='l2',
                        help='loss mode')


    return parser.parse_args()



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    log_file = os.path.join(LOGFILEPATH , 'trainlog.txt')
    log_init(log_file)

    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    

    if args.model == 'UNet':
        net = UNet(n_channels=1, n_classes=1, bilinear=True)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

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
                  model=args.model)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)