"""eval script"""
import torch
import torch.nn.functional as F
from tqdm import  tqdm
from utils.loss_util import cal_polynomial_model_loss, cal_radial_model_loss




def eval_net(net, loader, device, model, batch_size, order, dist_model):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False, ncols=80) as pbar:
        for batch in loader:
            imgs, corner = batch['image'], batch['corner']
            imgs = imgs.to(device=device, dtype=torch.float32)
            corner = corner.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                net_out = net(imgs)
                parameters = net_out['parameters']
                if model == 'UNet_half':
                    heatmap = None
                else:
                    heatmap = net_out['heatmap']
            if dist_model == 'poly':
                tot += cal_polynomial_model_loss(batch_size, order, parameters, corner, device)
            elif dist_model == 'radi':
                tot += cal_radial_model_loss(batch_size, order, parameters, corner, device)
            pbar.update()

    net.train()
    return tot / n_val , heatmap
