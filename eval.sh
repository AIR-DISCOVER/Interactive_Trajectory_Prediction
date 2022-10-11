OUTPUT_DIR=../additional_files_for_m2i/vv; \
DATA_DIR=../waymo_dataset/validation_interactive/; \
DATA_TXT=/DATA2/lpf/baidu/M2I_2/baidu_dataset_for_m2i_vv/valid_data.txt; \
RELATION_GT_DIR=../additional_files_for_m2i/validation_interactive_gt_relations.pickle; \
CUDA_VISIBLE_DEVICES=7 python -m src.run --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --config conditional_pred.yaml \
--relation_file_path ${RELATION_GT_DIR} \
--future_frame_num 80 \
-e \
--eval_exp_path eval_exp_path \
--debug_mode --model_recover_path 7
# --visualize
