import numpy as np
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    # Load Dataset
    data = pickle.load(open(r'20220920.pkl','rb'))
    print(data.keys())
    x_data = data['state/future/x']
    y_data = data['state/future/y']
    length_data = data['state/future/length']
    width_data = data['state/future/width']
    traj_num = len(data['state/id'])
    a = data['state/future/timestamp_micros']
    print(x_data.shape)
    '''
    # There are multiple trajectories 13
    for i in range(traj_num):
        traj_x = x_data[i,:]
        traj_y = y_data[i,:]
        plt.scatter(traj_x, traj_y)
        plt.savefig(str(i)+'.png')
        plt.clf()
    '''