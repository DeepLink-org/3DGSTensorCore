DATA=[your_data_path]
SCENE=[your_scene_name]
CKPT=[your_checkpoint_path]

# export CUDA_VISIBLE_DEVICES=0

python render.py \
    -s ${DATA}/${SCENE}/ \
    -m ${CKPT}/${SCENE}/ \
    --eval 