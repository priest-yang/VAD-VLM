
echo "Start train sample data !!!!!!"

torchrun \
    --nproc_per_node=1 \
    --master_port=28509 \
    ./lwad/adbase/vad/train.py \
    ./data/sample_data/VAD_sample_config.py \
    --launcher=pytorch \
    --deterministic


echo "Start convert gt map !!!!!!"

python lwad/convert_gt_map_json.py \
    --data_root ./data/sample_data/nuplan/dataset \
    --pkl_path  ./data/sample_data/sample_ann.pkl \
    --save_path ./data/sample_data/eval_map.json

echo "Start inference !!!!!!"

torchrun \
    --nproc_per_node=1 \
    --master_port=29303 \
    ./lwad/adbase/vad/test.py  ./data/sample_data/VAD_sample_config.py \
    ./work_dirs/VAD_sample_config/latest.pth \
    --json_dir='./data/sample_data/test_result' \
    --launcher=pytorch \
    --eval=bbox

