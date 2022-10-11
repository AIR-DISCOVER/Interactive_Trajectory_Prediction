import argparse
import itertools
import logging
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm as tqdm_

from . import utils, structs, globals
from .modeling.vectornet import VectorNet
from .waymo_tutorial import _parse
from .dataset_waymo import get_instance


import yaml

import pickle
import copy
from IPython import embed

def init_config(args):
    '''from utils.py'''
    if args.config is not None:
        fs = open(os.path.join('.', 'configs', args.config))
        config = yaml.load(fs, yaml.FullLoader)
        for key in config:
            setattr(args, key, config[key])

    dic = {}
    for i, param in enumerate(args.other_params + args.eval_params + args.train_params):
        if '=' in param:
            index = str(param).index('=')
            key = param[:index]
            value = param[index + 1:]
            # key, value = param.split('=')
            dic[key] = value if not str(value).isdigit() else int(value)
        else:
            dic[param] = True
    args.other_params = dic
    # print(dict(sorted(vars(args_).items())))

    os.makedirs(args.log_dir, exist_ok=True)

def init_m2i_args():
    args = argparse.Namespace()
    args.do_train = False
    args.do_eval = True
    args.do_test = False

    ################
    # args.data_dir = ''
    # args.data_txt = ''
    # args.output_dir = ''
    # args.temp_file_dir = ''
    # args.train_batch_size = 1
    # args.eval_batch_size = 1
    # args.model_recover_path = ''
    # args.validation_model = 0
    # args.learning_rate = 0
    # args.weight_decay = 0
    # args.num_train_epochs = 0
    # args.seed = 0
    ################
    args.log_dir = '/DATA2/lpf/baidu/additional_files_for_m2i/test4infer'
    args.no_cuda = False
    args.hidden_size = 128
    args.hidden_dropout_prob = 0.1
    args.sub_graph_depth = 3
    args.global_graph_depth = 1
    args.debug = False
    args.initializer_range = 0.02
    args.sub_graph_batch_size = 4096
    args.distributed_training = 0
    args.cuda_visible_device_num = None
    args.use_map = False
    args.reuse_temp_file = False
    args.old_version = False
    args.max_distance = 50.0
    args.no_sub_graph = False
    args.no_agents = False
    args.other_params = []
    args.eval_params = []
    args.train_params = []
    args.not_use_api = False
    args.core_num = 16
    args.visualize = False
    args.train_extra = False
    args.use_centerline = False
    args.autoregression = None
    args.lstm = False
    args.add_prefix = None
    args.attention_decay = False
    args.placeholder = 0.0
    args.multi = None
    args.method_span = [0, 1]
    args.nms_threshold = 7.2
    args.stage_one_K = None
    args.master_port = '12355'
    args.gpu_split = 0
    args.waymo = True
    args.argoverse = False
    args.nuscenes = False
    args.future_frame_num = 80
    args.future_test_frame_num = 16
    args.single_agent = True
    args.agent_type = 'vehicle'
    args.inter_agent_types = None
    args.mode_num = 6
    args.joint_target_each = 80
    args.joint_target_type = 'no'
    args.joint_nms_type = 'and'
    args.debug_mode = True
    args.traj_loss_coeff = 1.0
    args.short_term_loss_coeff = 0.0
    args.classify_sub_goals = False
    args.config = 'conditional_pred.yaml'
    args.relation_file_path = ''
    args.influencer_pred_file_path = None
    args.relation_pred_file_path = None
    args.reverse_pred_relation = False
    args.eval_rst_saving_number = None
    args.eval_exp_path = 'eval_exp_path'
    args.infMLP = 8
    args.relation_pred_threshold = 0.8
    args.direct_relation_path = None
    args.all_agent_ids_path = None
    args.vehicle_r_pred_threshold = None
    
    init_config(args)

    utils.args = args

    return args

def init_m2i(ckpt_path):
    args = init_m2i_args()

    model = VectorNet(args).cuda()
    model.eval()
    model_recover = torch.load(ckpt_path)
    model.load_state_dict(model_recover)

    return args, model

def choose_reactor(input_data):
    '''生成从车'''
    # influencer
    assert (input_data['state/is_sdc'] == 1).sum() == 1
    influencer_idx = (input_data['state/is_sdc'] == 1).nonzero()[0][0]

    # reactor
    valid_threshold_past = 1
    valid_threshold_current = 1
    temp_valid_past = input_data['state/past/valid'].copy().sum(axis=-1) >= valid_threshold_past
    temp_valid_current = input_data['state/current/valid'].copy().sum(axis=-1) >= valid_threshold_current
    chosen_cars = np.logical_and(temp_valid_past, temp_valid_current)   # use logical_and to ensure current is valid. get_instance() requires this.
    chosen_cars_idx = chosen_cars.nonzero()[0]

    split_data_list = {}
    for reactor_idx in chosen_cars_idx:
        if influencer_idx == reactor_idx:
            continue

        # label the cars
        new_decoded_example = {}
        new_decoded_example.update(copy.deepcopy(input_data))
        new_decoded_example['state/objects_of_interest'] *= 0
        new_decoded_example['state/objects_of_interest'][influencer_idx] = 1
        new_decoded_example['state/objects_of_interest'][reactor_idx] = 1
        new_decoded_example['state/tracks_to_predict'] *= 0
        new_decoded_example['state/tracks_to_predict'][influencer_idx] = 1
        new_decoded_example['state/tracks_to_predict'][reactor_idx] = 1

        # move influencer and reactor to the first 2 place
        for k,v in new_decoded_example.items():
            if k.split('/')[0] == 'state':
                if len(v.shape) == 2:
                    new_decoded_example[k][[influencer_idx,0],:] = v[[0,influencer_idx],:]
                    if reactor_idx == 0:
                        temp_reactor_idx = influencer_idx
                    else:
                        temp_reactor_idx = reactor_idx
                    new_decoded_example[k][[temp_reactor_idx,1],:] = v[[1,temp_reactor_idx],:]
                else:
                    new_decoded_example[k][[influencer_idx,0]] = v[[0,influencer_idx]]
                    if reactor_idx == 0:
                        temp_reactor_idx = influencer_idx
                    else:
                        temp_reactor_idx = reactor_idx
                    new_decoded_example[k][[temp_reactor_idx,1]] = v[[1,temp_reactor_idx]]

        # relation
        interaction_label = 1 if influencer_idx > reactor_idx else 0
        new_decoded_example['relation'] = np.array([influencer_idx, reactor_idx, interaction_label, 1])

        # filter out empty data to save memory space
        pastcurrent_valid = np.concatenate([new_decoded_example['state/past/valid'],new_decoded_example['state/current/valid']], axis=1)
        pastcurrent_valid_agent = pastcurrent_valid.sum(1)>0

        if not pastcurrent_valid_agent[0] == True and pastcurrent_valid_agent[1] == True:
            embed()
        assert pastcurrent_valid_agent[0] == True and pastcurrent_valid_agent[1] == True

        for k,v in new_decoded_example.items():
            if k.split('/')[0] == 'state':
                new_decoded_example[k] = new_decoded_example[k][pastcurrent_valid_agent]

        new_decoded_example['influencer_reactor_idx'] = (influencer_idx, reactor_idx)
        split_data_list[reactor_idx] = copy.deepcopy(new_decoded_example)

    return split_data_list, influencer_idx

def prepare_data(args, input_data):
    '''生成m2i所需格式'''
    inputs, decoded_example = _parse(input_data)
    sample_is_valid = inputs['sample_is_valid']
    tracks_to_predict = tf.boolean_mask(inputs['tracks_to_predict'], sample_is_valid)
    predict_agent_num = tracks_to_predict.numpy().sum()
    tracks_type = tf.boolean_mask(decoded_example['state/type'], sample_is_valid)
    tracks_type = tracks_type.numpy().copy().reshape(-1)
    instance = []
    for select in range(predict_agent_num):
        # Make sure both agents are of the same type if it is specified.
        # if type_is_ok(tracks_type[select], args) and type_is_ok(tracks_type[1 - select], args):
        if True:
            t = get_instance(args, inputs, decoded_example, str(decoded_example['influencer_reactor_idx'][0])+'_'+str(decoded_example['influencer_reactor_idx'][1]),
                                select=select)
            # change to a soft check when eval
            if t is not None:
                instance.append(t)

    assert len(instance) == 1, len(instance)
    return instance

def get_metric_params(mapping, args):
    gt_trajectory = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'gt_trajectory'))
    gt_is_valid = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'gt_is_valid'))
    object_type = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'object_type'))
    scenario_id = utils.get_from_mapping(mapping, 'scenario_id')[0]
    object_id = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'object_id'))

    if 'joint_eval' in args.other_params:
        assert len(mapping) == 2, len(mapping)
        gt_trajectory = gt_trajectory[tf.newaxis, :]
        gt_is_valid = gt_is_valid[tf.newaxis, :]
        object_type = object_type[tf.newaxis, :]

    return (
        gt_trajectory,
        gt_is_valid,
        object_type,
        scenario_id,
        object_id
    )

def run_m2i_inference(args, m2i_model, input_data):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    import tensorflow as tf
    global tf

    eval_prediction = False
    if eval_prediction:
        from .waymo_tutorial import MotionMetrics, metrics_config, metric_names
        motion_metrics = MotionMetrics(metrics_config)
        if args.mode_num != 6:
            motion_metrics.not_compute = True
        utils.motion_metrics = MotionMetrics(metrics_config)
        utils.metric_names = metric_names

    split_data_list, influencer_idx = choose_reactor(input_data)

    predicted_traj_x = np.ones(input_data['state/future/x'].shape) * -1.
    predicted_traj_x[influencer_idx] = copy.deepcopy(input_data['state/future/x'])[influencer_idx]
    predicted_traj_y = np.ones(input_data['state/future/y'].shape) * -1.
    predicted_traj_y[influencer_idx] = copy.deepcopy(input_data['state/future/y'])[influencer_idx]

    for k,v in split_data_list.items():
        modified_data_item = prepare_data(args, v)

        if eval_prediction:
            metric_params = get_metric_params(modified_data_item, args)

        pred_trajectory, pred_score, _ = m2i_model(modified_data_item, device)

        if eval_prediction:
            motion_metrics.args = args
            motion_metrics.update_state(
                tf.convert_to_tensor(pred_trajectory.astype(np.float32)),
                tf.convert_to_tensor(pred_score.astype(np.float32)),
                *metric_params
            )

        predicted_traj_x[k] = pred_trajectory[0,0,:,0]  # batch, highest score, time, x
        predicted_traj_y[k] = pred_trajectory[0,0,:,1]

        # print(k)

    if eval_prediction:
        print('all metric_values', len(motion_metrics.get_all()[0]))
        print(utils.metric_values_to_string(motion_metrics.result(), metric_names))

    return predicted_traj_x, predicted_traj_y

def main():
    '''test'''
    ckpt_path = '/DATA2/lpf/baidu/additional_files_for_m2i/test/model_save/model.12.bin'
    args, m2i_model = init_m2i(ckpt_path)

    input_data = np.load('/DATA2/lpf/baidu/M2I-main/trajactory/test.npy', allow_pickle=True).item()
    input_data.pop('relation')

    predicted_traj_x, predicted_traj_y = run_m2i_inference(args, m2i_model, input_data)

if __name__ == '__main__':
    main()

