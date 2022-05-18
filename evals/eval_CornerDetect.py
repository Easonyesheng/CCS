"""eval script"""
import torch
import torch.nn.functional as F
from tqdm import  tqdm
from models.model_utils import SuperPointNet_process



def eval_net(net, loader, device, model):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    heatmap_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False, ncols=80) as pbar:
        for batch in loader:
            imgs, true_heatmaps = batch['image'], batch['heatmap']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_heatmaps = true_heatmaps.to(device=device, dtype=heatmap_type)

            with torch.no_grad():
                net_out = net(imgs)
                
                if model == 'UNet':
                    heatmap_pred = net_out
            # torch.nn.functional.mse_loss
            tot += F.mse_loss(heatmap_pred, true_heatmaps).item()
            
            pbar.update()

    net.train()
    return tot / n_val, heatmap_pred
