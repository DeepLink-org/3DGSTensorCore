DATA=[your_data_path]
SCENE=[your_scene]
CKPT=[your_ckpt_path]

# export CUDA_VISIBLE_DEVICES=0

python train.py \
    -s ${DATA}/${SCENE}/ \
    -m ${CKPT}/${SCENE}/ 