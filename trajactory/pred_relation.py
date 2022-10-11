import numpy as np
import math


# predict relation between cars
# cars: list, indexs of cars
# x: numpy array (n, T)
# y: numpy array (n, T)
# width: numpy array (n, T)
# length: numpy array (n, T)
# delta: float, to compute distance of relation
# interval: int
# T: int
def pred_relation(cars, x, y, width, length, delta_d, delta_t, interval, T):
    delta = T - interval
    interval_num = int((x.shape[1] - T)/ delta) + 2
    infos = []
    for i in range(interval_num):
        d_min = 255.
        info = {}
        
        start_step = i * delta
        end_step = start_step + T  
        car0, car1 = cars

        if start_step >= x.shape[1] -T:
            start_step = x.shape[1] -T
            end_step = x.shape[1]

        info["car0"] = car0
        info["car1"] = car1
        info["start_step"] = start_step
        info["end_step"] = end_step
        
        car0_x_split = x[car0, start_step:end_step]
        car0_y_split = y[car0, start_step:end_step]
        car0_w_split = width[car0, start_step:end_step]
        car0_l_split = length[car0, start_step:end_step]

        car1_x_split = x[car1, start_step:end_step]
        car1_y_split = y[car1, start_step:end_step]
        car1_w_split = width[car1, start_step:end_step]
        car1_l_split = length[car1, start_step:end_step]


        for i in range(T):
            car0_x = car0_x_split[i]
            car0_y = car0_y_split[i]
        
            for j in range(T):
                car1_x = car1_x_split[j]
                car1_y = car1_y_split[j]

                d = get_distance(car0_x, car1_x, car0_y, car1_y)
                if d < d_min:
                    d_min = d
                    info["tao0"] = i
                    info["tao1"] = j
        # d_min is prepared, compute relation
        i = info["tao0"]
        j = info["tao1"]
        if d_min < (car0_w_split[i] + car1_w_split[j]) / 2 + delta_d:
            t0 = i * TIMESTEP
            t1 = j * TIMESTEP
            if np.abs(t0 - t1) < delta_t:
                # is relative
                info["t0"] = t0
                info["t1"] = t1
                if t0 > t1:
                    info["yield"] = car0
                    info["pass"] = car1
                elif t0 <= t1:
                    info["yield"] = car1
                    info["pass"] = car0
                info["d"] = d_min
            else:
                # no relation
                t0 = -1
                t1 = -1
                info["t0"] = t0
                info["t1"] = t1
        else:
            # no relation
            t0 = -1
            t1 = -1  
            info["t0"] = t0
            info["t1"] = t1
        infos.append(info)
    
    return infos
     
def get_distance(x0, x1, y0, y1):
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

if __name__ == '__main__':
    TIMESTEP = 0.14
    x = np.load("x.npy")
    y = np.load("y.npy")
    width = np.load("width.npy")
    length = np.load("length.npy")
    interval = 50
    delta_d = 0.2
    delta_t = 10
    T = 91
    cars = [10,11]
    infos = pred_relation(cars, x, y, width, length, delta_d, delta_t, interval, T)
