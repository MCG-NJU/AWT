
DESC_PATH=./descriptions/image_datasets
testset=$1

# number of descriptions per class
N_DESC=50

python -u evaluate.py \
        --test_set ${testset} \
        --arch ViT-B/16 \
        --descriptor_path ${DESC_PATH} \
        --num_descriptor ${N_DESC}