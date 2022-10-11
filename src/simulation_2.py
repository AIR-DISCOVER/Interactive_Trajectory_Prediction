import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import pathlib
import pickle
import cubic_spline_planner
import copy

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.1, 0.01])  # input cost matrix
Rd = np.diag([0.1, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 1.0])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 4.0  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 8.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 80.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.1  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 100.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -50.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

XY_GOAL_TOLERANCE = 5.0 # [m]

show_animation = True

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
    
class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(v, phi, delta):

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def update_state(state, a, delta):

    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC contorl with updating operational point iteraitvely
    """

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        if oa is not None and od is not None:
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= DU_TH:
                break
    else:
        pass
        #print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov


def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    elif prob.status == cvxpy.OPTIMAL_INACCURATE:
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False

def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def get_switch_back_course(dl, ax, ay):
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def list2mat(data, target_length, descending_order_index):
    # 填补数据
    for i in range(len(data)):
        while(len(data[i])<target_length):
            data[i] = list(data[i])
            data[i].append(data[i][-1])
        while(len(data[i])>target_length):
            data[i] = list(data[i])
            data[i].pop()
    
    original_order_data = [0] * len(data)
    for i in range(len(data)):
        original_order_data[descending_order_index[-1-i]] = data[i]

    # 封装成numpy矩阵
    for i in range(len(original_order_data)):
        # 第一个数据
        if(i==0):
            mat_data = np.asarray(original_order_data[i])
        else:
            mat_data = np.vstack((mat_data, original_order_data[i]))
    return mat_data

def calc_v(distance, v):
    if(v<=9.0):
        detect_range = 9.0
    else:
        detect_range = v
    obstacle_distance_range = detect_range #meter
    if(distance<=obstacle_distance_range):
        return (obstacle_distance_range - distance)
    else:
        return 0

def mpc_forward(original_data):
    dl = 1.0  # course tick
    # 加载数据集
    data = copy.deepcopy(original_data)
    x_data = data['state/future/x']
    y_data = data['state/future/y']
    length_data = data['state/past/length']
    width_data = data['state/past/width']
    traj_num = len(data['state/id'])
    descending_order_index = data['dummy/roadgraph_samples/distance_descending_order']

    # 给m2i下一轮输入
    m2i_data = data
    m2i_data['state/future/bbox_yaw'] = []
    m2i_data['state/future/x'] = []
    m2i_data['state/future/y'] = []
    m2i_data['state/future/vel_yaw'] = []
    m2i_data['state/future/velocity_x'] = []
    m2i_data['state/future/velocity_y'] = []

    xy = []
    # 每一时刻的障碍物坐标记录
    obs_cache = []

    for abba in range(traj_num):
        try:
            plt.clf()
            i = descending_order_index[-1-abba]
            progressBar(abba, traj_num, str(abba + 1) + '/' + str(traj_num) + ' | ' + "Running MPC")
            wp_x = x_data[i,:]
            wp_y = y_data[i,:]
            cx, cy, cyaw, ck = get_switch_back_course(dl, wp_x, wp_y)
            
            sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
            
            initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
            
            goal = [cx[-1], cy[-1]]

            state = initial_state

            # initial yaw compensation
            if state.yaw - cyaw[0] >= math.pi:
                state.yaw -= math.pi * 2.0
            elif state.yaw - cyaw[0] <= -math.pi:
                state.yaw += math.pi * 2.0

            time = 0.0
            x = [state.x]
            y = [state.y]
            yaw = [state.yaw]
            v = [state.v]
            t = [0.0]
            d = [0.0]
            a = [0.0]
            
            target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
            
            odelta, oa = None, None
            
            ai, di = 0, 0
            
            cyaw = smooth_yaw(cyaw)
            
            index = 0
            reached_goal = 0

            state_future_vel_yaw = []
            state_future_velocity_x = []
            state_future_velocity_y = []
            
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

            while MAX_TIME > time:
                state_future_vel_yaw.append(state.v/average_length/math.tan(di))
                state_future_velocity_x.append(state.v*math.cos(state.yaw))
                state_future_velocity_y.append(state.v*math.sin(state.yaw))

                plt.cla()
                xref, target_ind, dref = calc_ref_trajectory(
                    state, cx, cy, cyaw, ck, sp, dl, target_ind)

                x0 = [state.x, state.y, state.v, state.yaw]  # current state

                oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
                    xref, x0, dref, oa, odelta)

                if odelta is not None:
                    di, ai = odelta[0], oa[0]
                else:
                    mpc_solved = 0
                    break

                # 计算障碍物距离以及方向单位向量
                distances = []
                orientations = []
                for obs_list in obs_cache:
                    obs_x_list = obs_list[0]
                    obs_y_list = obs_list[1]
                    if(index<len(obs_x_list)):
                        obs_x = obs_x_list[index]
                        obs_y = obs_y_list[index]
                        if(show_animation):
                            plt.scatter(obs_x, obs_y, s=100)
                        distance = math.sqrt((state.x -obs_x)**2+(state.y -obs_y)**2)
                        distances.append(distance)
                        orientations.append([(obs_x - state.x)/distance, (obs_y - state.y)/distance])
                    else:
                        try:
                            obs_x = obs_x_list[-1]
                            obs_y = obs_y_list[-1]
                            if(show_animation):
                                plt.scatter(obs_x, obs_y, s=100)
                            distance = math.sqrt((state.x -obs_x)**2+(state.y -obs_y)**2)
                            distances.append(distance)
                            orientations.append([(obs_x - state.x)/distance, (obs_y - state.y)/distance])
                        except:
                            pass

                # 人工势场法被动避障
                force = [0, 0]
                for dd in range(len(distances)):
                    distance = distances[dd]
                    vv = calc_v(distance, state.v)
                    force[0] += vv*orientations[dd][0]
                    force[1] += vv*orientations[dd][1]

                if(reached_goal == 0):
                    # 通过人工势场法进行被动避障
                    if(force[0] != 0 or force[1] != 0):
                        pf_vx = -force[0]
                        pf_vy = -force[1]
                        mpc_vx = (state.v+ai*DT)*math.cos(state.yaw)
                        mpc_vy = (state.v+ai*DT)*math.sin(state.yaw)
                        vx = pf_vx + mpc_vx
                        vy = pf_vy + mpc_vy
                        if(show_animation):
                            plt.plot([state.x, state.x+vx], [state.y, state.y+vy],c='g')
                        vv = math.sqrt(vx**2 + vy**2)
                        psi = pi_2_pi(np.arctan2(vy, vx))
                        car_psi = pi_2_pi(state.yaw)
                        delta_psi = pi_2_pi(psi - car_psi)
                        u0 = vv*math.cos(delta_psi)
                        u1 = di + delta_psi
                        #ai = (u0-state.v)/DT
                        ai_part = 0.97
                        ai = ai*ai_part + (u0-state.v)/DT*(1-ai_part)
                        if(ai<0):
                            u1 = -u1
                        di = u1
                else:
                    ai = -state.v/DT
                    di = 0
                    # 通过人工势场法进行被动避障
                    if(force[0] != 0 or force[1] != 0):
                        pf_vx = -force[0]
                        pf_vy = -force[1]
                        vx = pf_vx
                        vy = pf_vy
                        if(show_animation):
                            plt.plot([state.x, state.x+vx], [state.y, state.y+vy],c='g')
                        vv = math.sqrt(vx**2 + vy**2)
                        psi = pi_2_pi(np.arctan2(vy, vx))
                        car_psi = pi_2_pi(state.yaw)
                        delta_psi = pi_2_pi(psi - car_psi)
                        u0 = vv*math.cos(delta_psi)
                        u1 = di + delta_psi
                        ai = (u0-state.v)/DT
                        if(ai<0):
                            u1 = -u1
                        di = u1

                state = update_state(state, ai, di)
                time = time + DT

                x.append(state.x)
                y.append(state.y)
                yaw.append(state.yaw)
                v.append(state.v)
                t.append(time)
                d.append(di)
                a.append(ai)

                if check_goal(state, goal, target_ind, len(cx)):
                    reached_goal = 1
                    #print("Goal")
                
                if(math.sqrt((state.x - goal[0])**2+(state.y - goal[1])**2) < XY_GOAL_TOLERANCE):
                    reached_goal = 1
                    #print("Goal")

                if show_animation:  # pragma: no cover
                    #plt.cla()
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect('key_release_event',
                            lambda event: [exit(0) if event.key == 'escape' else None])
                    if ox is not None:
                        plt.plot(ox, oy, "xr", label="MPC")
                    plt.plot(cx, cy, "-r", label="course")
                    plt.plot(x, y, "ob", label="trajectory")
                    plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
                    plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                    plot_car(state.x, state.y, state.yaw, steer=di)
                    plt.axis("equal")
                    plt.grid(True)
                    plt.title("Time[s]:" + str(round(time, 2))
                            + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
                    plt.pause(0.0001)
                
                index += 1
            xy = [x,y]
            obs_cache.append(xy)

            # 把轨迹数据储存到返回值m2i_data
            m2i_data['state/future/x'].append(x)
            m2i_data['state/future/y'].append(y)
            m2i_data['state/future/bbox_yaw'].append(yaw)
            m2i_data['state/future/vel_yaw'].append(state_future_vel_yaw)
            m2i_data['state/future/velocity_x'].append(state_future_velocity_x)
            m2i_data['state/future/velocity_y'].append(state_future_velocity_y)
        except:
            print('index: '+str(abba)+' has unknown problem')
            print('大概率是轨迹不能拟合')
            xy = [wp_x, wp_y]
            obs_cache.append(xy)
            data = copy.deepcopy(original_data)
            m2i_data['state/future/x'].append(data['state/future/x'][i,:].tolist())
            m2i_data['state/future/y'].append(data['state/future/y'][i,:].tolist())
            m2i_data['state/future/bbox_yaw'].append(data['state/future/bbox_yaw'][i,:].tolist())
            m2i_data['state/future/vel_yaw'].append(data['state/future/vel_yaw'][i,:].tolist())
            m2i_data['state/future/velocity_x'].append(data['state/future/velocity_x'][i,:].tolist())
            m2i_data['state/future/velocity_y'].append(data['state/future/velocity_y'][i,:].tolist())

    m2i_data['state/future/x'] = list2mat(m2i_data['state/future/x'], len(wp_x), descending_order_index)
    m2i_data['state/future/y'] = list2mat(m2i_data['state/future/y'], len(wp_x), descending_order_index)
    m2i_data['state/future/bbox_yaw'] = list2mat(m2i_data['state/future/bbox_yaw'], len(wp_x), descending_order_index)
    m2i_data['state/future/vel_yaw'] = list2mat(m2i_data['state/future/vel_yaw'], len(wp_x), descending_order_index)
    m2i_data['state/future/velocity_x'] = list2mat(m2i_data['state/future/velocity_x'], len(wp_x), descending_order_index)
    m2i_data['state/future/velocity_y'] = list2mat(m2i_data['state/future/velocity_y'], len(wp_x), descending_order_index)

    return m2i_data

def save_pkl(pkl_data,file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(pkl_data, f)

if __name__ == '__main__':
    original_data = pickle.load(open(r'20220923.pkl','rb'))
    m2i_data = mpc_forward(original_data)
    save_pkl(m2i_data, '20220925_mpc_output.pkl')