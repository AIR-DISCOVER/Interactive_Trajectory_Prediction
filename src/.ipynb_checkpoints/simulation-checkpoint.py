import copy

from unicodedata import ucd_3_2_0
from map import Map, Obstacle
import numpy as np
from reference_path import ReferencePath
from spatial_bicycle_models import BicycleModel
import matplotlib.pyplot as plt
from MPC import MPC
from scipy import sparse
import json
import codecs
import sys
import os
import math
import pickle
import pathlib
from CubicSpline import cubic_spline_planner

def progressBar(i, max, text):
    """
    Print a progress bar during training.
    :param i: index of current iteration/epoch.
    :param max: max number of iterations/epochs.
    :param text: Text to print on the right of the progress bar.
    :return: None
    """
    bar_size = 60
    j = (i + 1) / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()


def calc_v(distance):
    obstacle_distance_range = 15 #meter
    if(distance<=obstacle_distance_range):
        return 0.1*(obstacle_distance_range - distance)
    else:
        return 0

def list2mat(data, target_length):
    # print(target_length)
    # 填补数据
    for i in range(len(data)):
        while(len(data[i])<target_length):
            data[i] = list(data[i])
            data[i].append(data[i][-1])
        while(len(data[i])>target_length):
            data[i] = list(data[i])
            data[i].pop()

    # 封装成numpy矩阵
    for i in range(len(data)):
        # 第一个数据
        if(i==0):
            mat_data = np.asarray(data[i])
        else:
            mat_data = np.vstack((mat_data, data[i]))
    return mat_data

def angle_filter(angle):
    while(angle>=math.pi):
        angle = angle - 2*math.pi
    while(angle<=-math.pi):
        angle = angle + 2*math.pi
    return angle

def mpc_forward(original_data):
    # 加载数据集
    data = copy.deepcopy(original_data)
    x_data = data['state/future/x']
    y_data = data['state/future/y']
    length_data = data['state/past/length']
    width_data = data['state/past/width']
    traj_num = len(data['state/id'])
    descending_order_index = data['dummy/roadgraph_samples/distance_descending_order']

    # 加载地图文件
    map_data = Map(file_path='src/maps/5000_5000.png', origin=[-50, -50], resolution=0.1)

    # 每一时刻的障碍物坐标记录
    obs_cache = []

    # 给m2i下一轮输入
    m2i_data = data
    m2i_data['state/future/bbox_yaw'] = []
    m2i_data['state/future/x'] = []
    m2i_data['state/future/y'] = []
    m2i_data['state/future/vel_yaw'] = []
    m2i_data['state/future/velocity_x'] = []
    m2i_data['state/future/velocity_y'] = []
    
    # 遍历所有轨迹
    for abba in range(traj_num):
        i = descending_order_index[abba]
        progressBar(abba, traj_num, str(abba + 1) + '/' + str(traj_num) + ' | ' + "Running MPC")
        wp_x = x_data[i,:]
        wp_y = y_data[i,:]

        state_future_x = []
        state_future_y = []
        state_future_bbox_yaw = []
        state_future_vel_yaw = []
        state_future_velocity_x = []
        state_future_velocity_y = []

        # 轨迹数据非空
        if(len(wp_x) != 0):
            # 打印m2i输出waypoints点的个数
            #print('m2i output num points: '+str(len(wp_x)))

            # 判断mpc是否可以追踪
            flag = 1

            # 将输入的路径点按照分辨率进行插值，随后平滑处理
            reference_path = ReferencePath(map_data, wp_x, wp_y, resolution=0.1,
                                        smoothing_distance=5, max_width=30.0,
                                        circular=False)

            # 计算汽车的长
            average_length = 0
            available_length_num = 0
            for length in length_data[i]:
                if(length>0):
                    average_length += length
                    available_length_num += 1
            if(available_length_num == 0):
                average_length = 3
            else:
                average_length = average_length/available_length_num

            # 计算汽车的宽
            average_width = 0
            available_width_num = 0
            for width in width_data[i]:
                if(width>0):
                    average_width += width
                    available_width_num += 1
            if(available_width_num==0):
                average_width = 2
            else:
                average_width = average_width/available_width_num

            # 根据路径、地图数据，构建汽车模型
            try:
                car = BicycleModel(length=average_length, width=average_width,
                            reference_path=reference_path, Ts=0.2)
            except:
                #print("不出意外轨迹是一堆离散的点")
                #plt.clf()
                #plt.scatter(wp_x, wp_y)
                #plt.pause(0.5)
                flag = 0

            # 创建MPC求解器
            N = 20
            Q = sparse.diags([1.0, 1.0, 1.0])
            R = sparse.diags([0.5, 0.5])
            QN = sparse.diags([1.0, 1.0, 1.0])

            v_max = 6.0  # m/s
            delta_max = 0.66  # rad
            ay_max = 5.0  # m/s^2
            InputConstraints = {'umin': np.array([0.0, -np.tan(delta_max)/car.length]),
                                'umax': np.array([v_max, np.tan(delta_max)/car.length])}
            StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                                'xmax': np.array([np.inf, np.inf, np.inf])}
            mpc = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)
            # mpc控制器输入量，v和delta
            u = np.array([0, 0])
            # 根据约束信息，计算路径速度
            a_min = -0.1  # m/s^2
            a_max = 0.2  # m/s^2
            SpeedProfileConstraints = {'a_min': a_min, 'a_max': a_max,
                                    'v_min': 0.0, 'v_max': v_max, 'ay_max': ay_max}
            try:
                car.reference_path.compute_speed_profile(SpeedProfileConstraints)
            except:
                flag = 0
                continue

            # 仿真运行记时
            t = 0.0
            index = 0

            # 仿真运行轨迹记录
            x_cache = []
            y_cache = []
            xy_cache = []
            
            while (car.s < reference_path.length):
                # 记录修改的m2i数据
                state_future_bbox_yaw.append(car.temporal_state.psi)
                state_future_x.append(car.temporal_state.x)
                state_future_y.append(car.temporal_state.y)
                state_future_vel_yaw.append(u[1])
                state_future_velocity_x.append(u[0]*math.cos(car.temporal_state.psi))
                state_future_velocity_y.append(u[0]*math.sin(car.temporal_state.psi))

                goal_x = reference_path.waypoints[-1].x
                goal_y = reference_path.waypoints[-1].y
                goal_distance = math.sqrt((goal_x-car.temporal_state.x)**2+(goal_y-car.temporal_state.y)**2)
                if(goal_distance<=1.5):
                    #print('goal tolerance achieved!')
                    break

                if(t>=20):
                    break

                #plt.clf()
                # 记录汽车当前坐标
                x_cache.append(car.temporal_state.x)
                y_cache.append(car.temporal_state.y)

                # 计算障碍物距离以及方向单位向量
                distances = []
                orientations = []
                # 在地图上添加障碍物
                obs_to_map = []
                for obs_list in obs_cache:
                    obs_x_list = obs_list[0]
                    obs_y_list = obs_list[1]
                    if(index<len(obs_x_list)):
                        obs_x = obs_x_list[index]
                        obs_y = obs_y_list[index]
                        #plt.scatter(obs_x, obs_y, s=100)
                        #obs_to_map.append(Obstacle(cx=obs_x, cy=obs_y, radius=1))
                        distance = math.sqrt((car.temporal_state.x -obs_x)**2+(car.temporal_state.y -obs_y)**2)
                        distances.append(distance)
                        orientations.append([(obs_x - car.temporal_state.x)/distance, (obs_y - car.temporal_state.y)/distance])
                    else:
                        try:
                            obs_x = obs_x_list[-1]
                            obs_y = obs_y_list[-1]
                            #plt.scatter(obs_x, obs_y, s=100)
                            #obs_to_map.append(Obstacle(cx=obs_x, cy=obs_y, radius=1))
                            distance = math.sqrt((car.temporal_state.x -obs_x)**2+(car.temporal_state.y -obs_y)**2)
                            distances.append(distance)
                            orientations.append([(obs_x - car.temporal_state.x)/distance, (obs_y - car.temporal_state.y)/distance])
                        except:
                            pass

                # 更新地图
                #map_data.add_obstacles(obs_to_map)
                #car.reference_path.map = map_data

                # 人工势场法被动避障
                force = [0, 0]
                for d in range(len(distances)):
                    distance = distances[d]
                    v = calc_v(distance)
                    force[0] += v*orientations[d][0]
                    force[1] += v*orientations[d][1]
                    
                # 如果周围没有其他车辆
                if(force[0] == 0 and force[1] == 0):
                    try:
                        # 通过mpc求解控制量
                        u = mpc.get_control()
                    except:
                        # print("不出意外是锐角转弯")
                        flag = 0
                        break
                # 如果周围有其他车辆
                else:
                    
                    try:
                        # 通过mpc求解控制量
                        u = mpc.get_control()
                    except:
                        # print("不出意外是锐角转弯")
                        flag = 0
                        break
                    
                    
                    # 更新mpc类内部状态
                    #mpc.model.get_current_waypoint()
                    #mpc.model.spatial_state = mpc.model.t2s(reference_state=
                    #    mpc.model.temporal_state, reference_waypoint=
                    #    mpc.model.current_waypoint)
                    
                    # 通过人工势场法进行被动避障
                    pf_vx = -force[0]
                    pf_vy = -force[1]
                    mpc_vx = u[0]*math.cos(car.temporal_state.psi)
                    mpc_vy = u[0]*math.sin(car.temporal_state.psi)
                    vx = pf_vx + mpc_vx
                    vy = pf_vy + mpc_vy
                    v = math.sqrt(vx**2 + vy**2)
                    psi = np.arctan2(vy, vx)
                    #plt.plot([car.temporal_state.x, car.temporal_state.x+vx], [car.temporal_state.y, car.temporal_state.y+vy],c='g')
                    psi = angle_filter(psi)
                    #print(psi)
                    car_psi = angle_filter(car.temporal_state.psi)
                    delta_psi = psi - car_psi
                    delta_psi = angle_filter(delta_psi)
                    wz = delta_psi/car.Ts
                    if(wz>=delta_max):
                        wz = delta_max
                    elif(wz<=-delta_max):
                        wz = -delta_max
                    u0 = v*math.cos(delta_psi)
                    u1 = wz
                    if(u0<0):
                        u1 = -u1
                    u = np.array([u0, u1])
                #plt.plot([car.temporal_state.x, car.temporal_state.x+7*math.cos(car.temporal_state.psi)], [car.temporal_state.y, car.temporal_state.y+7*math.sin(car.temporal_state.psi)])

                # 汽车运动
                car.drive(u)
                # 记时
                t += car.Ts
                index += 1

                # 可视化路径
                #reference_path.show()
                # 可视化汽车
                #car.show()
                # 可视化mpc预测结果
                #mpc.show_prediction()

                # 清除障碍物
                map_data.clear_obstacles(obs_to_map)

                #plt.title('MPC Simulation: v(t): {:.2f}, delta(t): {:.2f}, Duration: '
                #    '{:.2f} s'.format(u[0], u[1], t))
                #plt.pause(0.01)
            
            # 整合记录轨迹
            xy_cache = [x_cache, y_cache]
            obs_cache.append(xy_cache)

            if(flag == 1):
                dx = np.diff(state_future_x)
                dy = np.diff(state_future_y)
                ds = np.hypot(dx, dy) # return math.sqrt(x**2 + y**2)
                s = [0]
                s.extend(np.cumsum(ds))
                ds = 0.0
                while(len(wp_x)*ds < s[-1]):
                    ds += 0.01
                #print('ds: '+str(ds))
                if(ds>0):
                    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(state_future_x, state_future_y, ds=ds)
                    #print('mpc output num points: '+str(len(cx)))

                    m2i_data['state/future/x'].append(cx)
                    m2i_data['state/future/y'].append(cy)
                    m2i_data['state/future/bbox_yaw'].append(cyaw)
                    cvx = np.diff(cx)
                    cvx = np.append(cvx,[cvx[-1]])/car.Ts
                    cvy = np.diff(cy)
                    cvy = np.append(cvy,[cvy[-1]])/car.Ts
                    cvyaw = np.diff(cyaw)
                    cvyaw = np.append(cvyaw,[cvyaw[-1]])/car.Ts
                    m2i_data['state/future/vel_yaw'].append(cvyaw)
                    m2i_data['state/future/velocity_x'].append(cvx)
                    m2i_data['state/future/velocity_y'].append(cvy)
                else:
                    data = copy.deepcopy(original_data)
                    m2i_data['state/future/x'].append(data['state/future/x'][i,:].tolist())
                    m2i_data['state/future/y'].append(data['state/future/y'][i,:].tolist())
                    m2i_data['state/future/bbox_yaw'].append(data['state/future/bbox_yaw'][i,:].tolist())
                    m2i_data['state/future/vel_yaw'].append(data['state/future/vel_yaw'][i,:].tolist())
                    m2i_data['state/future/velocity_x'].append(data['state/future/velocity_x'][i,:].tolist())
                    m2i_data['state/future/velocity_y'].append(data['state/future/velocity_y'][i,:].tolist())
            else:
                data = copy.deepcopy(original_data)
                m2i_data['state/future/x'].append(data['state/future/x'][i,:].tolist())
                m2i_data['state/future/y'].append(data['state/future/y'][i,:].tolist())
                m2i_data['state/future/bbox_yaw'].append(data['state/future/bbox_yaw'][i,:].tolist())
                m2i_data['state/future/vel_yaw'].append(data['state/future/vel_yaw'][i,:].tolist())
                m2i_data['state/future/velocity_x'].append(data['state/future/velocity_x'][i,:].tolist())
                m2i_data['state/future/velocity_y'].append(data['state/future/velocity_y'][i,:].tolist())
                

    m2i_data['state/future/x'] = list2mat(m2i_data['state/future/x'], len(wp_x))
    m2i_data['state/future/y'] = list2mat(m2i_data['state/future/y'], len(wp_x))
    m2i_data['state/future/bbox_yaw'] = list2mat(m2i_data['state/future/bbox_yaw'], len(wp_x))
    m2i_data['state/future/vel_yaw'] = list2mat(m2i_data['state/future/vel_yaw'], len(wp_x))
    m2i_data['state/future/velocity_x'] = list2mat(m2i_data['state/future/velocity_x'], len(wp_x))
    m2i_data['state/future/velocity_y'] = list2mat(m2i_data['state/future/velocity_y'], len(wp_x))

    return m2i_data


if __name__ == "__main__":
    mpc_forward('sample_mpc_inputs.pkl')