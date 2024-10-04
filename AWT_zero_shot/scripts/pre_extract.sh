testsets=$1
data_root=/home/yuhan.zhu/play_coop/CoOp_dataset

# set visual augmented times
BS=50

python -u ./pre_extract.py ${data_root} \
        --test_set ${testsets} \
        --arch ViT-B/16 \
        --batch-size ${BS} \
        --seed 0