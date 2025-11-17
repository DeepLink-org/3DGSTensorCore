DATA=[your_data_path]
SCENE=/mnt/shared-storage-user/ailab-sys/liaozimu/data/tandt
CKPT=/mnt/shared-storage-user/ailab-sys/liaozimu/ckpts/gaussian_splatting/

# export CUDA_VISIBLE_DEVICES=0

python train.py \
    -s ${DATA}/${SCENE}/ \
    -m ${CKPT}/${SCENE}/ 