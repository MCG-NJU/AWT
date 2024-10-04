data_root=/home/yuhan.zhu/play_coop/CoOp_dataset

testsets=$1

python -u ./siglip.py ${data_root} \
        --test_set ${testsets} \
        --descriptor_path ./descriptions/image_datasets \
        --batch-size 50 \
        --num_descriptor 50