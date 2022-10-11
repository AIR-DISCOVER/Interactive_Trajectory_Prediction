from math import sqrt
import pickle5 as pickle
import os
import sys
import copy
import json
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
import cv2
from IPython import embed
from joblib import Parallel, delayed

def get_xml_file(loaded_inputs_for_common_road_path):
    suffix = loaded_inputs_for_common_road_path.split(".")[-1]
    assert (suffix == 'pkl' or suffix == 'pickle'), 'file type not is pkl or pickle!'
    with open(loaded_inputs_for_common_road_path, 'rb') as f:
        loaded_inputs_for_common_road = pickle.load(f)
    
    scenario_id = loaded_inputs_for_common_road['scenario/id'].split('#')[1]
    s_id = scenario_id.split("_")[0]
    video_name = loaded_inputs_for_common_road_path.split("/")[-1].split('.')[0]
    scenario_name = '/DATA2/lpf/baidu/M2I-main/data/baidu_dataset/yizhuang_json/yizhuang_hdmap' + s_id + '.json'
    output_path = '/DATA2/lpf/baidu/M2I-main/data/commonroad_dataset/output_trajectory/CNN_PEK-' + s_id
    obstacles_data_path = os.path.join('/DATA2/lpf/baidu/M2I-main/data/baidu_dataset/yizhuang/obstacles_tLights/yizhuang#' + s_id + '/yizhuang#' + scenario_id, 'obstacles.csv')
    obstacles = pd.read_csv(obstacles_data_path, sep=',')
    
    obstacles_id = loaded_inputs_for_common_road['state/id']
    is_sdc       = loaded_inputs_for_common_road['state/is_sdc']
    if suffix == 'pkl':
        point_x      = loaded_inputs_for_common_road['state/past/x']
        point_y      = loaded_inputs_for_common_road['state/past/y']
        length       = loaded_inputs_for_common_road['state/past/length']
        width        = loaded_inputs_for_common_road['state/past/width']
        bbox_yaw     = loaded_inputs_for_common_road['state/past/bbox_yaw']
        velocity_x   = loaded_inputs_for_common_road['state/past/velocity_x']
        velocity_y   = loaded_inputs_for_common_road['state/past/velocity_y'] 
    else:
        point_x      = loaded_inputs_for_common_road['state/x']
        point_y      = loaded_inputs_for_common_road['state/y']
        length       = loaded_inputs_for_common_road['state/length']
        width        = loaded_inputs_for_common_road['state/width']
        bbox_yaw     = loaded_inputs_for_common_road['state/bbox_yaw']
        velocity_x   = loaded_inputs_for_common_road['state/velocity_x']
        velocity_y   = loaded_inputs_for_common_road['state/velocity_y']
    
    for idx, value in enumerate(is_sdc):
        if value == 1:
            sdc_id = '6' + str(int(obstacles_id[idx]))
            break

    # scenario_len = point_x.shape[1]

    banchmark_id = 'CHN_PEK-' + s_id + '_1_T-1'
    file_name = 'CHN_PEK-' + s_id + '_1_T-1_' + str(int(sdc_id)) +  '.xml'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, file_name), 'w') as output_file:
        output_file.write('')
    with open(scenario_name, 'r') as f:
        lanelet_baidu_data_ori = json.load(f)
    lanelet_baidu_data_ori = lanelet_baidu_data_ori['LANE']

    map_center_x = obstacles['area_map.feature_position.x'].iloc[1]
    map_center_y = obstacles['area_map.feature_position.y'].iloc[1]
    # set show scope
    x_start = obstacles['perception_obstacle.position.x'].min() - 50
    x_end = obstacles['perception_obstacle.position.x'].max() + 50
    y_start = obstacles['perception_obstacle.position.y'].min() - 50
    y_end = obstacles['perception_obstacle.position.y'].max() + 50

    lane_type_baidu_list = ['BIKING', 'CITY_DRIVING', 'EMERGENCY_LANE','LEFT_TURN_WAITING_ZONE', 'ROUNDABOUT']
    lane_type_croad_list = ['bicycleLane', 'urban', 'driveWay', 'urban', 'urban']
    lanelet_baidu_data = process_lanelet_data(lanelet_baidu_data_ori, x_start, x_end, y_start, y_end)
    d_list = [d * ' ' for d in range(2, 14, 2)]  # 生成空格

    set_header(d_list, banchmark_id, output_path, file_name)

    set_lanelet(d_list, lanelet_baidu_data, lane_type_baidu_list, lane_type_croad_list, x_start, x_end, y_start, y_end, 
                    map_center_x, map_center_y, output_path, file_name)

    scenario_max_len = set_dynamic_obstacle_trajectory(d_list, obstacles_id, point_x, point_y, length, width, bbox_yaw, velocity_x, 
                    velocity_y, output_path, file_name)
    
    set_end(output_path, file_name)
    print("Everythind is saved in {}".format(os.path.join(output_path, file_name)))

    return os.path.join(output_path, file_name), scenario_max_len, video_name

def progressBar(i, max, text):
    bar_size = 60
    j = (i + 1) / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()

def process_lanelet_data(lanelet_baidu_data_ori, x_start, x_end, y_start, y_end):
    lanelet_baidu_data = copy.deepcopy(lanelet_baidu_data_ori)
    max_len = len(lanelet_baidu_data_ori)
    for i, lanelet_id in enumerate(lanelet_baidu_data_ori):
        left_boundary = lanelet_baidu_data_ori[lanelet_id]['left_boundary']
        left_boundary = [eval(x) for x in left_boundary]
        right_boundary = lanelet_baidu_data_ori[lanelet_id]['right_boundary']
        right_boundary = [eval(x) for x in right_boundary]
        # 选择区域坐标范围
        lane_len = min(len(left_boundary), len(right_boundary))
        flag_temp = 0
        for idx in range(lane_len):
            if left_boundary[idx][0] < x_start or left_boundary[idx][0] > x_end or left_boundary[idx][1] < y_start or left_boundary[idx][1] > y_end:
                flag_temp += 1
                continue
            if right_boundary[idx][0] < x_start or right_boundary[idx][0] > x_end or right_boundary[idx][1] < y_start or right_boundary[idx][1] > y_end:
                flag_temp += 1
                continue
        if flag_temp == lane_len or flag_temp == lane_len - 1:
            del[lanelet_baidu_data[lanelet_id]]
            # print("delete no content lanelet: {}...".format(lanelet_id))
            #progressBar(i, max_len, 'Remove invalid lanelets...')
    return lanelet_baidu_data
# set header
def set_header(d_list, banchmark_id, output_path, file_name):
    res = ''
    res += "<?xml version='1.0' encoding='UTF-8'?>" + '\n'
    res += '<commonRoad timeStepSize="' + str(0.1) + '" commonRoadVersion="2020a" author="AIR, Tsinghua University" affiliation="AIR, Tsinghua University" source="DAIR-V2X" benchmarkID="' + banchmark_id + '" date="2022-8-31">' + '\n'
    res +=      d_list[0] + '<location>' + '\n'
    res +=          d_list[1] + '<geoNameId>999</geoNameId>' + '\n'
    res +=          d_list[1] + '<gpsLatitude>999</gpsLatitude>' + '\n'
    res +=          d_list[1] + '<gpsLongitude>999</gpsLongitude>' + '\n'
    res +=      d_list[0] + '</location>' + '\n'
    res +=      d_list[0] + '<scenarioTags>' + '\n'
    res +=          d_list[1] + '<Intersection/>' + '\n'
    res +=          d_list[1] + '<Urban/>' + '\n'
    res +=      d_list[0] + '</scenarioTags>' + '\n'

    with open(os.path.join(output_path, file_name), 'a') as data_file:
        data_file.write(res)

def id_process(mode, id, lanelet_baidu_data=None):
    if mode == 'special':
        if id not in list(lanelet_baidu_data):
            return None
    id_list = re.split("_|-", id)
    tmp_str = ''
    for ch in id_list[0]:
        if ch.isalpha():
            ch = str(ord(ch) - ord('a'))
        tmp_str += ch
    res_id = tmp_str + ''.join(id_list[1:])

    # 简单应付一下id重复的问题，减小lanelet_id和obstacle的id一致的情况出现
    if len(res_id) < 8:
        res_id += (8 - len(res_id)) * '1'
    # 保证lanelet id的首字母为1
    res_id = '1' + res_id
    return res_id

def set_lanelet(d_list, lanelet_baidu_data, lane_type_baidu_list, lane_type_croad_list, x_start, x_end, y_start, y_end, map_center_x, map_center_y, output_path, file_name):
    max_len = len(lanelet_baidu_data)
    for index, lanelet_id in enumerate(lanelet_baidu_data):
        # print("set lanelet: ", lanelet_id)
        #progressBar(index, max_len, 'set lanelet...')
        res = ''
        leftBound = ''
        rightBound = ''
        lane_type      = lanelet_baidu_data[lanelet_id]['lane_type']
        left_boundary  = lanelet_baidu_data[lanelet_id]['left_boundary']
        left_boundary  = [eval(x) for x in left_boundary]
        right_boundary = lanelet_baidu_data[lanelet_id]['right_boundary']
        right_boundary = [eval(x) for x in right_boundary]
        predecessor    = lanelet_baidu_data[lanelet_id]['predecessors']
        successor      = lanelet_baidu_data[lanelet_id]['successors']
        adjacentLeft   = lanelet_baidu_data[lanelet_id]['l_neighbor_id']
        adjacentRight  = lanelet_baidu_data[lanelet_id]['r_neighbor_id']
        turn_direction = lanelet_baidu_data[lanelet_id]['turn_direction']
        intersection   = lanelet_baidu_data[lanelet_id]['is_intersection']
        lanelet_id     = id_process('normal', lanelet_id)

        res        += d_list[0] + '<lanelet id="' + lanelet_id + '">' + '\n'
        leftBound  +=      d_list[1] + '<leftBound>' + '\n'
        rightBound +=     d_list[1] + '<rightBound>' + '\n'

        lane_len = min(len(left_boundary), len(right_boundary))
        # flag_temp = 0
        for idx in range(lane_len):
            # 选择区域坐标范围
            if left_boundary[idx][0] < x_start or left_boundary[idx][0] > x_end or left_boundary[idx][1] < y_start or left_boundary[idx][1] > y_end:
                continue
            if right_boundary[idx][0] < x_start or right_boundary[idx][0] > x_end or right_boundary[idx][1] < y_start or right_boundary[idx][1] > y_end:
                continue
            leftBound +=          d_list[2] + '<point>' + '\n'
            leftBound +=              d_list[3] + '<x>' + str(left_boundary[idx][0] - map_center_x) + "</x>" + '\n'
            leftBound +=              d_list[3] + '<y>' + str(left_boundary[idx][1] - map_center_y) + "</y>" + '\n'
            leftBound +=          d_list[2] + '</point>' + '\n'
            
            rightBound +=          d_list[2] + '<point>' + '\n'
            rightBound +=              d_list[3] + '<x>' + str(right_boundary[idx][0] - map_center_x) + "</x>" + '\n'
            rightBound +=              d_list[3] + '<y>' + str(right_boundary[idx][1] - map_center_y) + "</y>" + '\n'
            rightBound +=          d_list[2] + '</point>' + '\n'

        leftBound  +=     d_list[1] + '</leftBound>' + '\n'
        rightBound +=     d_list[1] + '</rightBound>' + '\n'

        # 计算当前道路的方向(斜率表示)
        # lane_k = (left_boundary[0][1] - left_boundary[-1][1]) / (left_boundary[0][0] - left_boundary[-1][0])
        res += leftBound
        res += rightBound
        
        if len(predecessor) != 0:
            for i in range(len(predecessor)):
                predecessor_id = id_process('special', predecessor[i], lanelet_baidu_data=lanelet_baidu_data)
                if predecessor_id is not None:
                    res +=      d_list[1] + '<predecessor ref="' + predecessor_id + '"/>' + '\n'

        if len(successor) != 0: 
            for i in range(len(successor)):
                succersor_id = id_process('special', successor[i], lanelet_baidu_data=lanelet_baidu_data)
                if succersor_id is not None:
                    res +=      d_list[1] + '<successor ref="' + succersor_id + '"/>' + '\n'

        if adjacentLeft != 'None':
            adjacentLeft_id = id_process('special', adjacentLeft, lanelet_baidu_data=lanelet_baidu_data)
            if adjacentLeft_id is not None:
                # adjacentLeft_left_boundary = lanelet_baidu_data[adjacentLeft]['left_boundary']
                # adjacentLeft_left_boundary = [eval(x) for x in adjacentLeft_left_boundary]
                # adjacentLeft_lane_k = (adjacentLeft_left_boundary[0][1] - adjacentLeft_left_boundary[-1][1]) / (adjacentLeft_left_boundary[0][0] - adjacentLeft_left_boundary[-1][0])
                # if lane_k * adjacentLeft_lane_k >= 0:
                #     drivingDir = 'same'
                # else:
                #     drivingDir = 'opposite'
                res +=      d_list[1] + '<adjacentLeft ref="' + adjacentLeft_id + '" drivingDir="same"/>' + '\n'

        if adjacentRight != 'None':
            adjacentRight_id = id_process('special', adjacentRight, lanelet_baidu_data=lanelet_baidu_data)
            if adjacentRight_id is not None:
                # adjacentRight_left_boundary = lanelet_baidu_data[adjacentRight]['left_boundary']
                # adjacentRight_left_boundary = [eval(x) for x in adjacentRight_left_boundary]
                # adjacentRight_lane_k = (adjacentRight_left_boundary[0][1] - adjacentRight_left_boundary[-1][1]) / (adjacentRight_left_boundary[0][0] - adjacentRight_left_boundary[-1][0])
                # if lane_k * adjacentRight_lane_k >= 0:
                #     drivingDir = 'same'
                # else:
                #     drivingDir = 'opposite'
                res +=      d_list[1] + '<adjacentRight ref="' + adjacentRight_id + '" drivingDir="same"/>' + '\n'

        # idx = lane_type_baidu_list.index(lane_type_baidu_list)
        lane_type_ = lane_type_croad_list[lane_type_baidu_list.index(lane_type)]
        res +=      d_list[1] + '<laneletType>' + lane_type_ + '</laneletType>' + '\n'
        res += d_list[0] + '</lanelet>' + '\n'
        
        with open(os.path.join(output_path, file_name), 'a') as data_file:
            data_file.write(res)

def set_dynamic_obstacle_trajectory(d_list, obstacles_id, point_x, point_y, length, width, bbox_yaw, velocity_x, 
                    velocity_y, output_path, file_name):
    
    LEN0 = point_x.shape[0]
    LEN1 = point_x.shape[1]
    assert LEN0 > 3, "trajectory is too short..."
    valid_list_dict = {}
    scenario_max_len = 0

    # 获取动态物体有效的部分序列
    for i in range(LEN0):
        j = 0
        valid_list = []
        while j < LEN1 and point_x[i, j] == -1 and point_y[i, j] == -1:
            j += 1
        valid_list.append(j)
        while j < LEN1 and (point_x[i, j] != -1 or point_y[i, j] != -1):
            j += 1
        valid_list.append(j)
        valid_list_dict[i] = valid_list
        scenario_max_len = max(scenario_max_len, valid_list[1] - valid_list[0] + 1)
    
    velocity = -1 * np.ones((LEN0, LEN1), dtype=np.float32)
    acceleration = -1 * np.ones((LEN0, LEN1), dtype=np.float32)
    for i in range(LEN0):
        # print('get v and a: ', obstacles_id[i])
        #progressBar(i, LEN0, 'get v and a...')
        
        start_idx = valid_list_dict[i][0]
        end_idx = valid_list_dict[i][1]

        for j in range(start_idx, end_idx):
            velocity[i, j] = sqrt(velocity_x[i, j] ** 2 + velocity_y[i, j] ** 2)
            if j == start_idx:
                velocity_first_time = sqrt(velocity_x[i, j + 1] ** 2 + velocity_y[i, j + 1] ** 2)
                acceleration[i, j] = (velocity_first_time - velocity[i, j]) / 0.1
            elif j != end_idx - 1:
                v0 = sqrt((velocity_x[i, j - 1] ** 2 + velocity_y[i, j - 1] ** 2))
                v2 = sqrt((velocity_x[i, j + 1] ** 2 + velocity_y[i, j + 1] ** 2))
                acceleration[i, j] = (v2 - v0) / (0.1 * 2)
            else:
                v_n_1 = sqrt((velocity_x[i, j - 1] ** 2 + velocity_y[i, j - 1] ** 2))
                v_n_2 = sqrt((velocity_x[i, j - 2] ** 2 + velocity_y[i, j - 2] ** 2))
                acceleration[i, j] = (3 * velocity[i, j] + v_n_2 - 4 * v_n_1) / (0.1 * 2) 
                
    for idx_0, obstacle_id in enumerate(obstacles_id):
        start_idx = valid_list_dict[idx_0][0]
        end_idx = valid_list_dict[idx_0][1]
        if start_idx == end_idx or start_idx + 1 == end_idx:
            continue
        # print('process dynamicObstacle: obstacle_id')
        #progressBar(idx_0, obstacles_id.shape[0], 'process dynamicObstacle...')
        res = ''
        # set first time information
        res += d_list[0] + '<dynamicObstacle id="' + '6' + str(int(obstacle_id)) + '">' + '\n'
        res +=      d_list[1] + '<type>' + str('car') + '</type>' + '\n'
        # just for rectangle, If you have other shapes, modify here slightly
        res +=      d_list[1] + '<shape>' + '\n'
        res +=          d_list[2] + '<rectangle>' + '\n'
        res +=              d_list[3] + '<length>' + str(length[idx_0, start_idx]) + '</length>' + '\n'
        res +=              d_list[3] + '<width>' + str(width[idx_0, start_idx]) + '</width>' + '\n'
        res +=          d_list[2] + '</rectangle>' + '\n'
        res +=      d_list[1] + '</shape>' + '\n'
        res +=      d_list[1] + '<initialState>' + '\n'
        res +=          d_list[2] + '<position>' + '\n'
        res +=              d_list[3] + '<point>' + '\n'
        res +=                  d_list[4] + '<x>' + str(point_x[idx_0, start_idx]) + '</x>' + '\n'
        res +=                  d_list[4] + '<y>' + str(point_y[idx_0, start_idx]) + '</y>' + '\n'
        res +=              d_list[3] + '</point>' + '\n'
        res +=          d_list[2] + '</position>' + '\n'
        res +=          d_list[2] + '<orientation>' + '\n'
        res +=              d_list[3] + '<exact>' + str(bbox_yaw[idx_0, start_idx]) + '</exact>' + '\n'
        res +=          d_list[2] + '</orientation>' + '\n'
        res +=          d_list[2] + '<time>' + '\n'
        res +=              d_list[3] + '<exact>' + str(int(start_idx)) + '</exact>' + '\n'  # init state set time=0
        res +=          d_list[2] + '</time>' + '\n'
        res +=          d_list[2] + '<velocity>' + '\n'
        res +=              d_list[3] + '<exact>' + str(velocity[idx_0, start_idx]) + '</exact>' + '\n'
        res +=          d_list[2] + '</velocity>' + '\n'
        res +=          d_list[2] + '<acceleration>' + '\n'
        res +=              d_list[3] + '<exact>' + str(acceleration[idx_0, start_idx]) + '</exact>' + '\n'
        res +=          d_list[2] + '</acceleration>' + '\n'
        res +=      d_list[1] + '</initialState>' + '\n'
        res +=      d_list[1] + '<trajectory>' + '\n'

        for idx_1 in range(start_idx + 1, end_idx):
            res +=          d_list[2] + '<state>' + '\n'
            res +=              d_list[3] + '<position>' + '\n'
            res +=                  d_list[4] + '<point>' + '\n'
            res +=                      d_list[5] + '<x>' + str(point_x[idx_0, idx_1]) + '</x>' + '\n'
            res +=                      d_list[5] + '<y>' + str(point_y[idx_0, idx_1]) + '</y>' + '\n'
            res +=                  d_list[4] + '</point>' + '\n'
            res +=              d_list[3] + '</position>' + '\n'
            res +=              d_list[3] + '<orientation>' + '\n'
            res +=                  d_list[4] + '<exact>' + str(bbox_yaw[idx_0, idx_1]) + '</exact>' + '\n'
            res +=              d_list[3] + '</orientation>' + '\n'
            res +=              d_list[3] + '<time>' + '\n'
            res +=                  d_list[4] + '<exact>' + str(idx_1) + '</exact>' + '\n'
            res +=              d_list[3] + '</time>' + '\n'
            res +=              d_list[3] + '<velocity>' + '\n'
            res +=                  d_list[4] + '<exact>' + str(velocity[idx_0, idx_1]) + '</exact>' + '\n'
            res +=              d_list[3] + '</velocity>' + '\n'
            res +=              d_list[3] + '<acceleration>' + '\n'
            res +=                  d_list[4] + '<exact>' + str(acceleration[idx_0, idx_1]) + '</exact>' + '\n'
            res +=              d_list[3] + '</acceleration>' + '\n'
            res +=          d_list[2] + '</state>' + '\n'
        res +=      d_list[1] + '</trajectory>' + '\n'
        res += d_list[0] + '</dynamicObstacle>' + '\n' 

        with open(os.path.join(output_path, file_name), 'a') as data_file:
            data_file.write(res)
    
    return scenario_max_len

def set_end(output_path, file_name):
    res = ''
    res += '</commonRoad>'
    with open(os.path.join(output_path, file_name), 'a') as data_file:
            data_file.write(res)

def xml2video(dir_path, file_name, scenario_len, video_name):
    file_path = dir_path
    # video_name = re.split('/', file_path)[-1][:-4]
    dir_path_ = os.path.join('/DATA2/lpf/baidu/M2I-main/video/output_trajectory/data_img', file_name, video_name)
    if not os.path.exists(dir_path_):
        os.makedirs(dir_path_)
    for path in os.listdir(dir_path_):
        file_name_ = os.path.join(dir_path_, path)
        os.remove(file_name_)
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    sdc_id = re.split('/', file_path)[-1][:-4].split('_')[-1]
    dynamic_obstacle_sdc = scenario.obstacle_by_id(int(sdc_id))
    scenario.remove_obstacle(dynamic_obstacle_sdc)

    # Len = int(video_name.split('_')[-1])
    for i in range(0, scenario_len): 
    # for i in range(0, 1799):
        # print('saving frame ' + str(i))
        progressBar(i, scenario_len, 'saving frame...')
        plt.figure(figsize=(10, 10))
        rnd = MPRenderer()
        scenario.draw(rnd, draw_params={'time_begin': i, 'time_end': i,"dynamic_obstacle": {"show_label": False, "draw_icon": True, "draw_shape": True, 
        'vehicle_shape': {'occupancy':{'shape': {'rectangle': {'facecolor':'white'}}}}}})
        planning_problem_set.draw(rnd)
        dynamic_obstacle_sdc.draw(rnd, draw_params={'time_begin': i, 'time_end': i, 'dynamic_obstacle': {'show_label': False, 'draw_icon': True, "draw_shape": True, 
                    'vehicle_shape': {'occupancy':{'shape': {'rectangle': {'facecolor':'red'}}}}}})
        rnd.render()
        plt.savefig(os.path.join(dir_path_,'frame' + str(i) + '.png')) 


    image_ori = cv2.imread(dir_path_ + '/frame0.png')
    video_size = (image_ori.shape[1], image_ori.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_path = '/DATA2/lpf/baidu/M2I-main/video/output_trajectory/data_video'
    if not os.path.exists(os.path.join(video_path, file_name)):
        os.makedirs(os.path.join(video_path, file_name))
    video = cv2.VideoWriter(os.path.join(video_path, file_name, video_name + ".mp4"), fourcc, 10, video_size, True)
    list_file = os.listdir(dir_path_)
    list_file.sort(key=lambda x:int(float(x[5:-4])))
    max_len = len(list_file)
    for i, file in enumerate(list_file):
        filename = os.path.join(dir_path_, file)
        frame = cv2.imread(filename)
        progressBar(i, max_len, 'get video...')
        video.write(frame)
    video.release() 

def traj_visualization(path):
    center_name = path.split("/")[-1]
    pkl_files = os.listdir(path)
    for pkl_file in pkl_files:
        loaded_inputs_for_common_road_path = os.path.join(path, pkl_file)
        file_path, scenario_max_len, video_name = get_xml_file(loaded_inputs_for_common_road_path)
        xml2video(file_path, center_name, scenario_max_len, video_name)

if __name__ == '__main__':
    path = '/DATA2/lpf/baidu/M2I-main/data/output_trajectory_data/data/m2i_only'
    center_name = path.split("/")[-1]
    pkl_files = os.listdir(path)
    for pkl_file in pkl_files:
        loaded_inputs_for_common_road_path = os.path.join(path, pkl_file)
        file_path, scenario_max_len, video_name = get_xml_file(loaded_inputs_for_common_road_path)
        xml2video(file_path, center_name, scenario_max_len, video_name)