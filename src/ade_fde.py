import numpy as np

def ade_fde(pred_traj, gt_traj):
    '''
    pred_traj: n*2    n 2d points
    gt_traj: n*2
    '''

    dis = np.sqrt((pred_traj[:,0] - gt_traj[:,0]) ** 2 + (pred_traj[:,1] - gt_traj[:,1]) ** 2)

    ade = np.mean(dis)
    fde = dis[-1]

    return ade, fde

