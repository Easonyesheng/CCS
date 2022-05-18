import numpy as np
import torch




def cal_polynomial_model_loss(batch, order, parameters, corners, device, corner_size=480):
    """Need Coordinate Normalization! NO USE
    Args:
        batch: batch size
        order: the order of polynimial model: n
        parameters: torch - batch x (n+2)*(n+1) x 1 x 1
        corners: torch - batch x 2 x N x 3
    Returns:
        loss: \sum ((x_c -x_d)^2 + (y_c - y_d)^2)
    """
    corners_before = corners[:,0]
    corners_before = corners_before.reshape(-1,3) # size = [batch*Nx3]
    X_c = corners_before[:,0]
    Y_c = corners_before[:,1]
    X_c = X_c / 480 - 0.5
    Y_c = Y_c / 480 - 0.5

    corners_after = corners[:,1]
    corners_after = corners_after.reshape(-1,3) # size = [batch*Nx3]
    X_d = corners_after[:, 0]
    Y_d = corners_after[:, 1]
    X_d = X_d / corner_size - 0.5
    Y_d = Y_d / corner_size - 0.5

    parameters = parameters.reshape(-1,1)
    y_num = parameters.shape[0] // 2

    X_c_pred = torch.zeros(corners_before.shape[0], dtype=torch.float).to(device)
    Y_c_pred = torch.zeros(corners_before.shape[0], dtype=torch.float).to(device)
    

    index = 0
    for j in range(order+1):
        for i in range(0, order-j+1):
            X_c_pred += parameters[index] * torch.pow(X_d, i).mul(torch.pow(Y_d, order-j-i))
            Y_c_pred += parameters[index+y_num] * torch.pow(X_d, i).mul(torch.pow(Y_d, order-j-i))
            index += 1
    
    loss_l1 = torch.abs(X_c_pred-X_c) + torch.abs(Y_c_pred-Y_c)
    loss_l2 = torch.pow(X_c_pred-X_c,2) + torch.pow(Y_c_pred-Y_c,2) 
    loss = 0.8*torch.sum(loss_l1) + 0.2*torch.sum(loss_l2)

    return loss / 20

def cal_radial_model_loss(batch, order, parameters, corners, device, corner_size=480):
    """Calculate the loss in radial model
    Args:
        corners: [batchx3xNx2]
            corner_before = corners[batch, 0, :, :]
            corner_after = corners[batch, 1, :, :]
            para = corners[batch, order, 0, 0]
    """
    # corner_size = corners.to(device).type(torch.float32)
    loss_tot = 0
    for i in range(batch):
        corners_before = corners[i,0] # [Nx2]
        # corners_before = corners_before.reshape(-1,2) 
        X_c = corners_before[:,0]
        Y_c = corners_before[:,1]
        X_c = X_c / corner_size - 0.5
        Y_c = Y_c / corner_size - 0.5
        # print(f'X_c shape = {X_c.shape}')

        corners_after = corners[i,1]
        # corners_after = corners_after.reshape(-1,2) # size = [batch*Nx2] WRONG!
        X_d = corners_after[:, 0]
        Y_d = corners_after[:, 1]
        X_d = X_d / corner_size - 0.5
        Y_d = Y_d / corner_size - 0.5
        # print(f'X_d shape = {X_d.shape}')


        parameters_batch = parameters[i] 
        W = corner_size
        H = corner_size

        r = torch.pow(X_d, 2) + torch.pow(Y_d, 2)
        #==================for any order
        if order!=1:
            r = r**(0.5)
            # print(r)
            coff = 0

            for i in range(parameters_batch.shape[0]):
                coff += parameters_batch[i]*torch.pow(r,i)

            X_c_pred = coff*X_d
            Y_c_pred = coff*Y_d
        # #====================for one parameter
        else:
            coff = parameters_batch
            
            print(f'coff {coff} shape = {coff.shape}')

            X_c_pred = (1+coff*r)*X_d
            Y_c_pred = (1+coff*r)*Y_d
        # #====================for one parameter




        loss_l1 = torch.abs(X_c_pred-X_c) + torch.abs(Y_c_pred-Y_c)
        loss_l2 = torch.pow(X_c_pred-X_c,2) + torch.pow(Y_c_pred-Y_c,2) 
        loss = torch.sum(loss_l1) + torch.sum(loss_l2)
        loss_a = loss / corners_before.shape[0]

        loss_tot+=loss_a

    return loss_tot / batch

