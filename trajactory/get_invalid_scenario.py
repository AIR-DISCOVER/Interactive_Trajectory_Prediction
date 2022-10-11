import pandas as pd
import numpy as np
from math import sqrt
import os
from IPython import embed
import re
import json
import sys
import copy

def get_invalid_scenario(file_path, error_list):
    obstacle_path = os.path.join(file_path, 'obstacles.csv')
    obstacles = pd.read_csv(obstacle_path, sep=',')
    dynamicObstacle = obstacles[(obstacles['perception_obstacle.sub_type'] != 'UNKNOWN_UNMOVABLE') & (obstacles['perception_obstacle.sub_type'] != 'TRAFFICCONE')]
    max_error_len = 0
    error_idx = 0
    camera_timestamp =  list(dynamicObstacle['header.camera_timestamp'].drop_duplicates())
    bool_value = False
    for i in range(len(camera_timestamp)):
        if i != 0:
            if (camera_timestamp[i] - camera_timestamp[i - 1]) * 1e-9 > 10:
                error_idx = i
                max_error_len = max(max_error_len, (camera_timestamp[i] - camera_timestamp[i - 1]) * 1e-9)
                bool_value = True
    return bool_value, max_error_len, error_idx


def main():
    all_data_length = 0
    error_list = []

    dir_name = '/DATA1/liyang/M2I_2/obstacles_tLights'
    dirs = os.listdir(dir_name)
    file_name_list = ['yizhuang#1', 'yizhuang#2', 'yizhuang#4', 'yizhuang#5', 'yizhuang#7', 'yizhuang#8', 'yizhuang#11', \
            'yizhuang#12', 'yizhuang#13', 'yizhuang#14', 'yizhuang#15', 'yizhuang#16', 'yizhuang#17', 'yizhuang#20', \
                'yizhuang#24', 'yizhuang#25', 'yizhuang#27', 'yizhuang#28']
    for path_name in dirs:
        if path_name in file_name_list:
            paths = os.listdir(os.path.join(dir_name, path_name))
            for idx, file_name in enumerate(paths):
                all_data_length += 1
                file_path = os.path.join(dir_name, path_name, file_name)
                print(file_path)
                bool_value, max_error_len, error_idx = get_invalid_scenario(file_path, error_list)
                if bool_value:
                    error_list.append([file_name, max_error_len, error_idx])

    embed()

if __name__ == '__main__':
    main()