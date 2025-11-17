rlaunch \
    --mount=gpfs://gpfs1/ailab-sys:/mnt/shared-storage-user/ailab-sys \
    --gpu=1 \
    --cpu=32 \
    --memory=200000 \
    --charged-group=sys_gpu --private-machine=yes \
    -- bash

# export http_proxy=http://liaozimu:l030efpklLlFOgnattBM3GgXhsShfDeutEvlCYTwCXVG3REC1qicC9grdv9S@10.1.20.50:23128
# export https_proxy=http://liaozimu:l030efpklLlFOgnattBM3GgXhsShfDeutEvlCYTwCXVG3REC1qicC9grdv9S@10.1.20.50:23128
# export no_proxy=10.0.0.0/8,100.0.0.0/8,35.220.164.252/32,.pjlab.org.cn
