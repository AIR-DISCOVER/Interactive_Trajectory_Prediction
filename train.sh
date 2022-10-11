OUTPUT_DIR=../additional_files_for_m2i/vv; \
DATA_DIR=../waymo_dataset/validation_interactive/; \
DATA_TXT=/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv/train_data.txt; \
RELATION_GT_DIR=../additional_files_for_m2i/validation_interactive_gt_relations.pickle; \
CUDA_VISIBLE_DEVICES=5,6,7 python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 120 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type vehicle \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj pair_vv raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 3
