import os
import numpy as np

from IPython import embed

root_path = '/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv'
save_path = './all_data.txt'

all_path = []

map_list = sorted([f for f in os.listdir(root_path) if os.path.isdir(f)])
for map_item in map_list:
    map_item_path = os.path.join(root_path, map_item)
    
    scene_list = sorted([f for f in os.listdir(map_item_path)])
    for scene_item in scene_list:
        scene_path = os.path.join(map_item_path, scene_item)

        sub_scene_list = sorted([f for f in os.listdir(scene_path)])
        for sub_scene_item in sub_scene_list:
            all_path.append(os.path.join(scene_path, sub_scene_item))

np.savetxt(save_path, all_path, fmt='%s')
