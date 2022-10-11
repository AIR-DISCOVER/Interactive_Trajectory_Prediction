'''
按照m2i方式计算距离和时间来得到relation
'''
from sre_constants import JUMP
import pandas as pd
import numpy as np
from math import sqrt
import os
import json
import sys
import copy
import matplotlib.pyplot as plt
import time
import copy
import math

from scipy import interpolate

from IPython import embed
import tqdm

def gen_dataset(obstacles_path, traffic_lights_path, hdmap_path_path, output_scene_path, target_data_type):
    jump_step = 1

    obstacles = pd.read_csv(obstacles_path, sep=',')
    # traffic_ligths = pd.read_csv(traffic_lights_path, sep=',')

    map_center_x = obstacles['area_map.feature_position.x'].iloc[1]
    map_center_y = obstacles['area_map.feature_position.y'].iloc[1]
    map_center_z = obstacles['area_map.feature_position.z'].iloc[1]

    x_start = obstacles['perception_obstacle.position.x'].min()
    x_end = obstacles['perception_obstacle.position.x'].max()
    y_start = obstacles['perception_obstacle.position.y'].min()
    y_end = obstacles['perception_obstacle.position.y'].max()

    dynamicObstacle = obstacles[(obstacles['perception_obstacle.sub_type'] != 'UNKNOWN_UNMOVABLE') & (obstacles['perception_obstacle.sub_type'] != 'TRAFFICCONE')]
    staticObstacle = obstacles[(obstacles['perception_obstacle.sub_type'] == 'UNKNOWN_UNMOVABLE') | (obstacles['perception_obstacle.sub_type'] == 'TRAFFICCONE')]
    dynamicObstacle_baidu_list = ['CAR', 'VAN', 'CYCLIST', 'MOTORCYCLIST', 'PEDESTRIAN', 'TRICYCLIST', 'BUS', 'TRUCK']
    dynamicObstacle_croad_list = ['car', 'truck', 'bicycle', 'motorcycle', 'pedestrian', 'truck', 'bus', 'truck']

    sequence_num_start = 0
    time_list = np.unique(np.array(dynamicObstacle['header.camera_timestamp']))
    obstacles_len = len(time_list)
    sequence = {}
    for i in range(len(time_list)):
        key = time_list[i]
        sequence[key] = i

    dynamicObstacle_id = list(dynamicObstacle['perception_obstacle.id'].drop_duplicates())
    staticObstacle_id = list(staticObstacle['perception_obstacle.id'].drop_duplicates())
    with open(hdmap_path_path, 'r') as f:
        lanelet_baidu_data = json.load(f)
    lanelet_baidu_data = lanelet_baidu_data['LANE']

    def process_data(staticObstacle, dynamicObstacle, staticObstacle_id, dynamicObstacle_id):
        '''处理了被错误识别为static obstacle 的物体'''
        static_id = copy.deepcopy(staticObstacle_id)
        for d_id in static_id:
            if d_id in dynamicObstacle_id:
                print("Duplicate ID: {}...".format(d_id))
                sub_type = list(dynamicObstacle[dynamicObstacle['perception_obstacle.id'] == d_id]['perception_obstacle.sub_type'])[0]
                staticObstacle.loc[staticObstacle['perception_obstacle.id'] == d_id, 'perception_obstacle.sub_type'] = sub_type
                s_obstacle_data = staticObstacle[staticObstacle['perception_obstacle.id'] == d_id]
                # 移除staticObstacle_id中的id并删除staticObstacle中错误id所在的所有行
                staticObstacle_id.remove(d_id)
                staticObstacle = staticObstacle.drop(index=staticObstacle[(staticObstacle['perception_obstacle.id'] == d_id)].index.tolist())
                staticObstacle = staticObstacle.reset_index(drop=True)
                dynamicObstacle = dynamicObstacle.append(s_obstacle_data, ignore_index=True)
        return staticObstacle, dynamicObstacle, staticObstacle_id, dynamicObstacle_id

    def set_lanelet(lanelet_baidu_data):
        LaneType = {
            "NONE": 0,
            "HIGHWAY_DRIVING":1,
            "CITY_DRIVING" : 2,
            "SHOULDER" : 1,
            "BIKING" : 3,
            "SIDEWALK" : 4,
            "BORDER" : 15,
            "RESTRICTED" : 17,
            "PARKING" : 17,
            "BIDIRECTIONAL" : 2,
            "MEDIAN" : 16,
            "ROADWORKS" : 17,
            "TRAM" : 4,
            "TAIL" : 4,
            "ENTRY" : 4,
            "EXIT" : 4,
            "OFFRAMP" : 4,
            "ONRAMP" : 4,
            "LEFT_TURN_WAITING_ZONE" : 17,
            "EMERGENCY_LANE" : 2,
            "ROUNDABOUT" : 4,
            "REVERSIBLE" : 4,
            "DIRECTION_VARIABLE" : 2,
            "BUS_LANE" : 2,
            "TOLL_LANE" : 2,
            "NO_TURN_WAITING_ZONE" : 19}
        lane = 0
        _xyz = []
        _id = []
        _type = []
        cnt = []
        for lanelet_id in lanelet_baidu_data:
            centerline = lanelet_baidu_data[lanelet_id]['centerline']
            lane_type = lanelet_baidu_data[lanelet_id]['lane_type']
            # has_traffic_control = lanelet_baidu_data[lanelet_id]['has_traffic_control']
            # left_boundary = lanelet_baidu_data[lanelet_id]['left_boundary']
            # right_boundary = lanelet_baidu_data[lanelet_id]['right_boundary']
            xb = float(centerline[0].split(',')[0].split('(')[1])
            yb = float(centerline[0].split(',')[1].split(')')[0])
            xe = float(centerline[-1].split(',')[0].split('(')[1])
            ye = float(centerline[-1].split(',')[1].split(')')[0])
            if  (xb<x_start-20) or (yb<y_start-20) or (xe>x_end+20) or (ye>y_end+20):
                lane = lane + 1
                continue
            _xyz.extend(centerline)
            _id.extend((np.ones((len(centerline))) * lane).tolist())
            _type.extend((np.ones((len(centerline))) * float(LaneType[lane_type])).tolist())
            cnt.append(len(_xyz))
            lane = lane + 1
        dir = np.zeros((len(_xyz),3), dtype=np.float32)
        xyz = -1 * np.ones((len(_xyz),3), dtype=np.float32)
        for i in range(len(_xyz)):
            xyz[i,0] = float(_xyz[i].split(',')[0].split('(')[1]) - map_center_x
            xyz[i,1] = float(_xyz[i].split(',')[1].split(')')[0]) - map_center_y
            if i > 0:
                dir[i-1,0] = (xyz[i,0] - xyz[i-1,0]) / (sqrt((xyz[i,0] - xyz[i-1,0])**2+(xyz[i,1] - xyz[i-1,1])**2) + 1e-9)
                dir[i-1,1] = (xyz[i,1] - xyz[i-1,1]) / (sqrt((xyz[i,0] - xyz[i-1,0])**2+(xyz[i,1] - xyz[i-1,1])**2) + 1e-9)
                if i in cnt:
                    dir[i-1,0] = 0
                    dir[i-1,1] = 0

        id = (np.array(_id)-np.array(_id).min())[:,None]
        type = np.array(_type).astype(np.float)
        valid = np.ones((len(id), 1))

        # x = xyz[:,0]
        # y = xyz[:,1]
        # plt.figure(figsize=(50, 50))
        # plt.title("yizhuang_hdmap14.json")
        # plt.scatter(x, y, s=50, c='b')
        # plt.savefig('/DATA1/liyang/M2I_2/image/road4.png')
        
        roadgraph = {'roadgraph_samples/dir':dir, 'roadgraph_samples/id':id, 'roadgraph_samples/type':type, 
                    'roadgraph_samples/xyz':xyz, 'roadgraph_samples/valid':valid.astype(np.int64)}
        return roadgraph

    def set_obstacle(dynamicObstacle, dynamicObstacle_id, obstacles_len):
        LEN = obstacles_len
        obj = 0
        num = len(dynamicObstacle_id)
        bbox_yaw = -1 * np.ones((num,LEN), dtype=np.float32)
        length = -1 * np.ones((num,LEN), dtype=np.float32)
        height = -1 * np.ones((num,LEN), dtype=np.float32)
        width = -1 * np.ones((num,LEN), dtype=np.float32)
        timestamp_micros = -1 * np.ones((num,LEN))
        valid = np.zeros((num,LEN))
        vel_yaw = -1 * np.ones((num,LEN))
        velocity_x = -1 * np.ones((num,LEN), dtype=np.float32)
        velocity_y = -1 * np.ones((num,LEN), dtype=np.float32)
        x = -1 * np.ones((num,LEN), dtype=np.float32)
        y = -1 * np.ones((num,LEN), dtype=np.float32)
        z = -1 * np.ones((num,LEN), dtype=np.float32)
        id = -1 * np.ones((num))
        is_sdc = -1 * np.ones((num))
        objects_of_interest = 0 * np.ones((num), dtype=np.int64)
        tracks_to_predict = 0 * np.ones((num))
        type = -1 * np.ones((num))

        time_begin = dynamicObstacle['header.camera_timestamp'].min()
        time = time_list-time_begin
        time = np.round(time*1e-9, 3)

        for dObstacle_id in dynamicObstacle_id:
            # print(dObstacle_id)
            d_obstacle_data = dynamicObstacle[dynamicObstacle['perception_obstacle.id'] == dObstacle_id]
            if len(d_obstacle_data) < 3:  # If the number of outgoing frames is less than 3, remov it directly
                continue
            #######
            # maybe unnecessary
            d_obstacle_len_fake = d_obstacle_data['header.sequence_num'].max() - d_obstacle_data['header.sequence_num'].min()   # + 1
            d_obstacle_len_true = len(d_obstacle_data)
            if d_obstacle_len_fake > d_obstacle_len_true:
                continue
            #######

            sub_type = str(d_obstacle_data.iloc[0]['perception_obstacle.sub_type'])
            if sub_type not in dynamicObstacle_baidu_list:
                continue

            # 按照时间戳排序
            d_obstacle_data = d_obstacle_data.sort_values('header.camera_timestamp', ascending=True)
            d_obstacle_data = d_obstacle_data.reset_index(drop=True)

            sub_type = str(d_obstacle_data.iloc[0]['perception_obstacle.sub_type'])
            if sub_type not in dynamicObstacle_baidu_list:
                continue
            
            id[obj] = dObstacle_id
            if sub_type == 'UNKNOW':
                type[obj] = 0
            elif sub_type in ['CAR', 'VAN', 'BUS', 'TRUCK']:
                type[obj] = 1
            elif sub_type in ['CYCLIST', 'MOTORCYCLIST', 'TRICYCLIST']:
                type[obj] = 3
            elif sub_type == 'PEDESTRIAN':
                type[obj] = 2
            is_sdc[obj] = 0

            idx = dynamicObstacle_baidu_list.index(sub_type)
            sub_type = dynamicObstacle_croad_list[idx]

            # 在进行轨迹定义的时候，首先需要明确，此时的obstacle_data是从索引1开始的
            d_obstacle_data_len = len(np.array(d_obstacle_data['header.camera_timestamp']))
            start = d_obstacle_data.iloc[0]['header.camera_timestamp']
            j = (sequence[start]-sequence_num_start)
            for i in range(d_obstacle_data_len):
                if i<d_obstacle_data_len-1:
                    if d_obstacle_data.iloc[i]['header.camera_timestamp'] == d_obstacle_data.iloc[i+1]['header.camera_timestamp']:
                        continue
                x[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.position.x'] - float(map_center_x)
                y[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.position.y'] - float(map_center_y)
                z[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.position.z'] - float(map_center_z)
                timestamp_micros[obj,j] = d_obstacle_data.iloc[i]['header.sequence_num']
                height[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.height']
                width[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.width']
                length[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.length']
                valid[obj,j] = 1
                bbox_yaw[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.theta']
                velocity_x[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.velocity.x']
                velocity_y[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.velocity.y']
                vel_yaw[obj,j] = np.arctan(velocity_y[obj,j]/(velocity_x[obj,j]+1e-9))
                j = j+1
            obj = obj + 1

        # print('obj', obj)
        
        # 插值
        LEN2 = int(time.max()//0.1) + 1
        bbox_yaw_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        length_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        height_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        width_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        timestamp_micros_new = -1 * np.ones((num,LEN2))
        valid_new = np.zeros((num,LEN2))
        vel_yaw_new = -1 * np.ones((num,LEN2))
        velocity_x_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        velocity_y_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        x_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        y_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        z_new = -1 * np.ones((num,LEN2), dtype=np.float32)
        
        for i in range(x.shape[0]): # all car
            if valid[i].sum() == 0:
                continue
            valid_tick_min = valid[i].nonzero()[0].min()
            valid_tick_max = valid[i].nonzero()[0].max()
            assert valid[i].sum() == valid_tick_max - valid_tick_min + 1

            valid_time_min = time[valid_tick_min]
            valid_time_max = time[valid_tick_max]

            valid_new_tick_min = int(valid_time_min // 0.1) # extrapolate
            valid_new_tick_max = int(valid_time_max // 0.1)

            # prepare input and value
            func_input_list = time[valid_tick_min: valid_tick_max+1]
            bbox_yaw_func_value_list    = bbox_yaw[i][valid_tick_min: valid_tick_max+1]
            length_func_value_list      = length[i][valid_tick_min: valid_tick_max+1]
            height_func_value_list      = height[i][valid_tick_min: valid_tick_max+1]
            width_func_value_list       = width[i][valid_tick_min: valid_tick_max+1]
            timestamp_func_value_list   = timestamp_micros[i][valid_tick_min: valid_tick_max+1]
            valid_func_value_list       = valid[i][valid_tick_min: valid_tick_max+1]
            vel_yaw_func_value_list     = vel_yaw[i][valid_tick_min: valid_tick_max+1]
            vx_func_value_list          = velocity_x[i][valid_tick_min: valid_tick_max+1]
            vy_func_value_list          = velocity_y[i][valid_tick_min: valid_tick_max+1]
            x_func_value_list           = x[i][valid_tick_min: valid_tick_max+1]
            y_func_value_list           = y[i][valid_tick_min: valid_tick_max+1]
            z_func_value_list           = z[i][valid_tick_min: valid_tick_max+1]

            # fit function
            bbox_yaw_func   = interpolate.interp1d(func_input_list, bbox_yaw_func_value_list, fill_value='extrapolate')
            length_func     = interpolate.interp1d(func_input_list, length_func_value_list, fill_value='extrapolate')
            height_func     = interpolate.interp1d(func_input_list, height_func_value_list, fill_value='extrapolate')
            width_func      = interpolate.interp1d(func_input_list, width_func_value_list, fill_value='extrapolate')
            timestamp_func  = interpolate.interp1d(func_input_list, timestamp_func_value_list, fill_value='extrapolate')
            valid_func      = interpolate.interp1d(func_input_list, valid_func_value_list, fill_value='extrapolate')
            vel_yaw_func    = interpolate.interp1d(func_input_list, vel_yaw_func_value_list, fill_value='extrapolate')
            vx_func         = interpolate.interp1d(func_input_list, vx_func_value_list, fill_value='extrapolate')
            vy_func         = interpolate.interp1d(func_input_list, vy_func_value_list, fill_value='extrapolate')
            x_func          = interpolate.interp1d(func_input_list, x_func_value_list, fill_value='extrapolate')
            y_func          = interpolate.interp1d(func_input_list, y_func_value_list, fill_value='extrapolate')
            z_func          = interpolate.interp1d(func_input_list, z_func_value_list, fill_value='extrapolate')

            query_time_list = np.arange(valid_new_tick_min, (valid_new_tick_max+1), 1)*0.1
            assert len(query_time_list) == valid_new_tick_max - valid_new_tick_min + 1

            bbox_yaw_new[i][valid_new_tick_min:(valid_new_tick_max+1)]         = bbox_yaw_func(query_time_list)
            length_new[i][valid_new_tick_min:(valid_new_tick_max+1)]           = length_func(query_time_list)
            height_new[i][valid_new_tick_min:(valid_new_tick_max+1)]           = height_func(query_time_list)
            width_new[i][valid_new_tick_min:(valid_new_tick_max+1)]            = width_func(query_time_list)
            timestamp_micros_new[i][valid_new_tick_min:(valid_new_tick_max+1)] = timestamp_func(query_time_list)
            valid_new[i][valid_new_tick_min:(valid_new_tick_max+1)]            = valid_func(query_time_list)
            vel_yaw_new[i][valid_new_tick_min:(valid_new_tick_max+1)]          = vel_yaw_func(query_time_list)
            velocity_x_new[i][valid_new_tick_min:(valid_new_tick_max+1)]       = vx_func(query_time_list)
            velocity_y_new[i][valid_new_tick_min:(valid_new_tick_max+1)]       = vy_func(query_time_list)
            x_new[i][valid_new_tick_min:(valid_new_tick_max+1)]                = x_func(query_time_list)
            y_new[i][valid_new_tick_min:(valid_new_tick_max+1)]                = y_func(query_time_list)
            z_new[i][valid_new_tick_min:(valid_new_tick_max+1)]                = z_func(query_time_list)

        # bbox_yaw = tf.convert_to_tensor(bbox_yaw)
        trajectory = {'state/bbox_yaw':bbox_yaw_new, 'state/x':x_new, 'state/y':y_new, 'state/z':z_new, 
                        'state/height': height_new, 'state/width':width_new, 'state/length':length_new, 'state/vel_yaw':vel_yaw_new, 
                        'state/valid': valid_new, 'state/velocity_x':velocity_x_new, 'state/velocity_y':velocity_y_new, 
                        'state/timestamp_micros':timestamp_micros_new, 'state/id':id, 'state/is_sdc':is_sdc, 'state/type':type,
                        'state/objects_of_interest':objects_of_interest, 'state/tracks_to_predict':tracks_to_predict}

        # embed(header='1')
        return trajectory

    def set_traffic_lights(state_len):
        LEN = state_len
        # LEN = 1
        # traffic_size = 16
        traffic_size = 0
        id = -1 * np.ones((LEN,traffic_size))
        state = -1 * np.ones((LEN,traffic_size))
        valid = -1 * np.ones((LEN,traffic_size))
        x = -1 * np.ones((LEN,traffic_size), dtype=np.float32)
        y = -1 * np.ones((LEN,traffic_size), dtype=np.float32)
        z = -1 * np.ones((LEN,traffic_size), dtype=np.float32)

        traffic_light = {'traffic_light_state/id':id, 'traffic_light_state/state':state, 'traffic_light_state/valid':valid,
                        'traffic_light_state/x':x, 'traffic_light_state/y':y, 'traffic_light_state/z':z}
        
        return traffic_light
    
    def split_decoded_data(decoded_example):
        data_total_tick = decoded_example['state/x'].shape[1]

        past = 10
        current = 1
        future = 80
        single_all_tick = past + current + future
        overlap = 20
        delta = single_all_tick - overlap

        split_count = (data_total_tick - single_all_tick) // delta + 1

        decoded_example_group = []

        for i in range(split_count):
            new_decoded_example = {}

            begin_tick = i * delta
            past_end_tick = begin_tick + past
            current_end_tick = past_end_tick + current
            end_tick = begin_tick + single_all_tick
            assert end_tick == current_end_tick + future

            for k,v in decoded_example.items():
                if k == 'scenario/id':
                    new_decoded_example[k] = copy.deepcopy(decoded_example[k])
                elif len(v.shape) == 2:
                    if v.shape[0] == data_total_tick:
                        new_decoded_example[k.split('/')[0]+'/past/'+k.split('/')[1]] = decoded_example[k].copy()[begin_tick:past_end_tick, :]
                        new_decoded_example[k.split('/')[0]+'/current/'+k.split('/')[1]] = decoded_example[k].copy()[past_end_tick:current_end_tick, :]
                        new_decoded_example[k.split('/')[0]+'/future/'+k.split('/')[1]] = decoded_example[k].copy()[current_end_tick:end_tick, :]
                    elif v.shape[1] == data_total_tick:
                        new_decoded_example[k.split('/')[0]+'/past/'+k.split('/')[1]] = decoded_example[k].copy()[:, begin_tick:past_end_tick]
                        new_decoded_example[k.split('/')[0]+'/current/'+k.split('/')[1]] = decoded_example[k].copy()[:, past_end_tick:current_end_tick]
                        new_decoded_example[k.split('/')[0]+'/future/'+k.split('/')[1]] = decoded_example[k].copy()[:, current_end_tick:end_tick]
                    else:
                        new_decoded_example[k] = decoded_example[k].copy()
                else:
                    new_decoded_example[k] = decoded_example[k].copy()

            decoded_example_group.append(new_decoded_example)

        return decoded_example_group

    def get_relation(new_decoded_example):
        TIMESTEP = 0.1
        delta_d = 6
        delta_t = 10

        car0 = 0
        car1 = 1

        car0_state_id = int(new_decoded_example['state/id'][car0])
        car1_state_id = int(new_decoded_example['state/id'][car1])

        car0_x_split = new_decoded_example['state/future/x'][car0]
        car0_y_split = new_decoded_example['state/future/y'][car0]
        car0_w_split = new_decoded_example['state/future/width'][car0]

        car1_x_split = new_decoded_example['state/future/x'][car1]
        car1_y_split = new_decoded_example['state/future/y'][car1]
        car1_w_split = new_decoded_example['state/future/width'][car1]

        distance = np.sqrt((car0_x_split.reshape(-1,1) - car1_x_split)**2 + (car0_y_split.reshape(-1,1) - car1_y_split)**2)
        d_min = distance.min()
        i = (distance == d_min).nonzero()[0][0]
        j = (distance == d_min).nonzero()[1][0]

        # if d_min < (car0_w_split[i] + car1_w_split[j]) / 2 + delta_d:
        if d_min < delta_d:
            if (i < j and car0_state_id < car1_state_id) or (i > j and car0_state_id > car1_state_id):
                return_relation = np.array([car0_state_id, car1_state_id, 0, 1])
            else:
                return_relation = np.array([car0_state_id, car1_state_id, 1, 1])
            # print(return_relation[2])
        else:
            return_relation = np.array([car0_state_id, car1_state_id, 2, 1])

        return return_relation
        
    def get_distance(x0, x1, y0, y1):
        return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    def gen_relation_data_sample(decoded_example_item, target_data_type):
        '''根据valid情况选择主车/从车'''
        valid_threshold_past = 10
        valid_threshold_current = 1
        valid_threshold_future = 80
        temp_valid_past = decoded_example_item['state/past/valid'].copy().sum(axis=-1) >= valid_threshold_past
        temp_valid_current = decoded_example_item['state/current/valid'].copy().sum(axis=-1) >= valid_threshold_current
        temp_valid_future = decoded_example_item['state/future/valid'].copy().sum(axis=-1) >= valid_threshold_future
        valid_objs = np.logical_and(np.logical_and(temp_valid_past, temp_valid_current), temp_valid_future)

        obj_cars = decoded_example_item['state/type'] == 1
        obj_cycs = decoded_example_item['state/type'] == 3
        obj_peds = decoded_example_item['state/type'] == 2
    
        chosen_cars = np.logical_and(valid_objs, obj_cars)
        chosen_cars_idx = chosen_cars.nonzero()[0]
        chosen_cycs = np.logical_and(valid_objs, obj_cycs)
        chosen_cycs_idx = chosen_cycs.nonzero()[0]
        chosen_peds = np.logical_and(valid_objs, obj_peds)
        chosen_peds_idx = chosen_peds.nonzero()[0]

        inf_objs_idx = chosen_cars_idx
        if target_data_type == 'vv':
            rec_objs_idx = chosen_cars_idx
            agent_pair_label = 1
        elif target_data_type == 'vp':
            rec_objs_idx = chosen_peds_idx
            agent_pair_label = 2
        elif target_data_type == 'vc':
            rec_objs_idx = chosen_cycs_idx
            agent_pair_label = 3
        else:
            raise NotImplementedError

        data_sample_list_inter_temp = []
        data_sample_list_noninter_temp = []
        for influcner_idx in inf_objs_idx:
            for reactor_idx in rec_objs_idx:
                if influcner_idx == reactor_idx:
                    continue
            
                # label the cars
                new_decoded_example = {}
                new_decoded_example.update(copy.deepcopy(decoded_example_item))
                new_decoded_example['state/is_sdc'][influcner_idx] = 1
                new_decoded_example['state/objects_of_interest'][influcner_idx] = 1
                new_decoded_example['state/objects_of_interest'][reactor_idx] = 1
                new_decoded_example['state/tracks_to_predict'][influcner_idx] = 1
                new_decoded_example['state/tracks_to_predict'][reactor_idx] = 1

                # move influencer and reactor to the first 2 place
                for k,v in new_decoded_example.items():
                    if k.split('/')[0] == 'state':
                        if len(v.shape) == 2:
                            new_decoded_example[k][[influcner_idx,0],:] = v[[0,influcner_idx],:]
                            if reactor_idx == 0:
                                temp_reactor_idx = influcner_idx
                            else:
                                temp_reactor_idx = reactor_idx
                            new_decoded_example[k][[temp_reactor_idx,1],:] = v[[1,temp_reactor_idx],:]
                        else:
                            new_decoded_example[k][[influcner_idx,0]] = v[[0,influcner_idx]]
                            if reactor_idx == 0:
                                temp_reactor_idx = influcner_idx
                            else:
                                temp_reactor_idx = reactor_idx
                            new_decoded_example[k][[temp_reactor_idx,1]] = v[[1,temp_reactor_idx]]

                # relation
                return_relation = get_relation(new_decoded_example)
                new_decoded_example['relation'] = return_relation

                # filter out empty data to save memory space
                pastcurrent_valid = np.concatenate([new_decoded_example['state/past/valid'],new_decoded_example['state/current/valid']], axis=1)
                pastcurrent_valid_agent = pastcurrent_valid.sum(1)>0

                if not pastcurrent_valid_agent[0] == True and pastcurrent_valid_agent[1] == True:
                    embed()
                assert pastcurrent_valid_agent[0] == True and pastcurrent_valid_agent[1] == True

                for k,v in new_decoded_example.items():
                    if k.split('/')[0] == 'state':
                        new_decoded_example[k] = new_decoded_example[k][pastcurrent_valid_agent]

                if return_relation[2] == 2:
                    data_sample_list_noninter_temp.append(new_decoded_example)
                else:
                    data_sample_list_inter_temp.append(new_decoded_example)

        return data_sample_list_noninter_temp, data_sample_list_inter_temp

    staticObstacle, dynamicObstacle, staticObstacle_id, dynamicObstacle_id = process_data(staticObstacle, dynamicObstacle, staticObstacle_id, dynamicObstacle_id)
    roadgraph = set_lanelet(lanelet_baidu_data)
    trajectory = set_obstacle(dynamicObstacle, dynamicObstacle_id, obstacles_len)
    traffic_light = set_traffic_lights(trajectory['state/x'].shape[1])


    decoded_example = roadgraph.copy()
    decoded_example.update(trajectory)
    decoded_example = decoded_example.copy()
    decoded_example.update(traffic_light)
    scenario = output_scene_path.split('/')[-2] + '_' + output_scene_path.split('/')[-1]
    decoded_example['scenario/id'] = scenario
    
    decoded_example_group = split_decoded_data(decoded_example)

    final_data_sample_list_noninter = []
    final_data_sample_list_inter = []
    for decoded_example_item in decoded_example_group:
        data_sample_list_noninter_temp, data_sample_list_inter_temp = gen_relation_data_sample(decoded_example_item, target_data_type)
        # print(len(data_sample_list_temp))
        final_data_sample_list_noninter.extend(data_sample_list_noninter_temp)
        final_data_sample_list_inter.extend(data_sample_list_inter_temp)

    np.random.shuffle(final_data_sample_list_noninter)
    np.random.shuffle(final_data_sample_list_inter)

    final_data_sample_list = []
    final_data_sample_list.extend(final_data_sample_list_noninter[:500])
    final_data_sample_list.extend(final_data_sample_list_inter[:750])

    print('len of final data sample list', len(final_data_sample_list))
    
    # save
    np.random.shuffle(final_data_sample_list)
    file_length = 200
    file_count = 5
    final_data_sample_list = final_data_sample_list[:min(file_length*file_count, len(final_data_sample_list))]

    relation_count = np.array([0,0,0])
    for item in final_data_sample_list:
        relation_count[int(item['relation'][2])]+=1
    print(relation_count)


    # file_count = len(final_data_sample_list) // file_length + 1
    file_count = len(final_data_sample_list) // file_length # drop last
    for i in range(file_count):
        begin_idx = i*file_length
        end_idx = min((i+1)*file_length, len(final_data_sample_list))
        output_scene_file_path = os.path.join(output_scene_path, '%03d.npy'%i)
        np.save(output_scene_file_path, final_data_sample_list[begin_idx:end_idx])

    return 0



def get_scenario_max_time_delta(obstacles_path):
    obstacles = pd.read_csv(obstacles_path, sep=',')
    dynamicObstacle = obstacles[(obstacles['perception_obstacle.sub_type'] != 'UNKNOWN_UNMOVABLE') & (obstacles['perception_obstacle.sub_type'] != 'TRAFFICCONE')]
    camera_timestamp =  list(dynamicObstacle['header.camera_timestamp'].drop_duplicates())

    max_time_delta = 0
    for i in range(len(camera_timestamp)):
        if i != 0:
            time_delta = (camera_timestamp[i] - camera_timestamp[i - 1]) * 1e-9
            max_time_delta = max(max_time_delta, time_delta)

    return max_time_delta



def main():
    target_data_type = 'vv'
    # prepare
    dataset_path = '/DATA1/liyang/M2I_2/obstacles_tLights'
    hdmap_json_path = '/DATA1/liyang/M2I_2/yizhuang_json'
    output_dataset_path = '/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_relation_'+target_data_type
    if not os.path.exists(output_dataset_path):
        os.makedirs(output_dataset_path)

    whole_map_path_list = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                                    if os.path.isdir(os.path.join(dataset_path, f))])
    scene_path_list = []

    hdmap_path_dict = {}

    for whole_map_path in whole_map_path_list:
        scene_path_list_temp = sorted([os.path.join(whole_map_path, f) for f in os.listdir(whole_map_path) 
                                        if os.path.isdir(os.path.join(whole_map_path, f))])
        scene_path_list.extend(scene_path_list_temp)

        hdmap_path = os.path.join(hdmap_json_path, 'yizhuang_hdmap'+str(int(whole_map_path.split('#')[1]))+'.json')
        for scene_path in scene_path_list_temp:
            hdmap_path_dict[scene_path] = hdmap_path

    # generate
    obstacles_name = 'obstacles.csv'
    traffic_lights_name = 'traffic_lights.csv'

    time_delta_invalid_threshold = 10 # 若场景中超过10s的跳变，则舍弃此场景

    for i in range(len(scene_path_list)):
        scene_path = scene_path_list[i]
        time1 = time.time()
        # input
        obstacles_path = os.path.join(scene_path, obstacles_name)
        traffic_lights_path = os.path.join(scene_path, traffic_lights_name)
        hdmap_path_path = hdmap_path_dict[scene_path]

        # check data
        max_time_delta = get_scenario_max_time_delta(obstacles_path)
        if max_time_delta > time_delta_invalid_threshold:
            print('>>>', scene_path, 'invalid scene. max time delta:', max_time_delta)
            continue

        # output
        output_scene_path = os.path.join(output_dataset_path, scene_path.split('/')[-2], scene_path.split('/')[-1])
        if not os.path.exists(output_scene_path):
            os.makedirs(output_scene_path)
        else:
            time2 = time.time()
            print('>>>', scene_path, 'time:', time2-time1)
            continue

        final_data_sample_list = gen_dataset(obstacles_path, traffic_lights_path, hdmap_path_path, output_scene_path, target_data_type)

        time2 = time.time()
        print('>>>', scene_path, 'time:', time2-time1)


if __name__ == '__main__':
    main()



