# Interactive Trajectory Prediction

This repository contains the codes for the interactive trajectory prediction part of our interactive simulator. The possible future trajectories of the non-ego vehicles can be predicted using these codes. We take M2I as the code base.

## Environment
Requires:

* Python 3.6
* PyTorch 1.6+

Install packages into a Conda environment (Cython, tensorflow, waymo-open-dataset, etc.):

``` bash
conda env create -f conda.cuda111.yaml
conda activate M2I
```

## Data Generation
Given the raw data collected from yizhuang, you can setup a dataset folder in the following hierarchy.
```
└── dataset
    ├── yizhuang_json
        ├── yizhuang_hdmap1.json
        ├── yizhuang_hdmap2.json
        ...
    └── obstacles_tLights
        ├── yizhuang#1
            ├── yizhuang#1_1
                ├── obstacles.csv
                └── traffic_lights.csv
            ├── yizhuang#1_2
            ...
        ├── yizhuang#2
        ...
```

Then you can assign `dataset_path` and `hdmap_json_path` in file **trajectory/gen_dataset.py** to be the path of obstacles_tLights and yizhuang_json, respectively. Specifying `output_dataset_path` as the output path, you can use the following command to generate the processed data for neural network training and validation.
```bash
cd trajectory
python gen_dataset.py
```

The processed data is organized in the following way.
```
└── output_dataset_name
    ├── yizhuang#1
        ├── yizhuang#1_1
            ├── 000.npy
            ├── 001.npy
            ...
        ├── yizhuang#1_2
        ...
    ├── yizhuang#2
    ...
```

Each npy file contains around 200 training/validation samples.

You can use the file **trajectory/gen_data_split.py** to generate the dataset information file and manually split the resulting .txt file into a training one and a validation one.

## Training
After generating training data, you can use the following command to train the conditional predictor.
```bash
OUTPUT_DIR=../additional_files_for_m2i/vv; \
DATA_DIR=/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv; \
DATA_TXT=/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv/train_data.txt; \
RELATION_GT_DIR=/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv; \
CUDA_VISIBLE_DEVICES=5,6,7 python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 120 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type vehicle \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj pair_vv raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 3
```

You need to set `DATA_TXT` as the path to the generated information file of the training split and `OUTPUT_DIR` as the output path.

To filter the type of agents for reactors, change the value of `--agent_type`. To filter both the type of reactors and influencers, change the flag of `pair_vv`.

## Evaluation

Use the following command to evaluate the trained model.

```
OUTPUT_DIR=../additional_files_for_m2i/vv; \
DATA_DIR=/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv; \
DATA_TXT=/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv/valid_data.txt; \
RELATION_GT_DIR=/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv; \
CUDA_VISIBLE_DEVICES=7 python -m src.run --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --config conditional_pred.yaml \
--relation_file_path ${RELATION_GT_DIR} \
--future_frame_num 80 \
-e \
--eval_exp_path eval_exp_path \
--debug_mode --model_recover_path 7
# --visualize
```

You need to set `DATA_TXT` as the path to the generated information file of the validation split and `OUTPUT_DIR` as the output path (the same as training output path). Use `--model_recover_path` to choose the trained model checkpoint of different epoch. Use `--visualize` to generate the visualization of predicted trajectories.

