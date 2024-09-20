args=(
    --dataset $3
    --ckpt vit_21k 
    --method synqt 
    --exp_name $2 
)
CUDA_VISIBLE_DEVICES=$1 python3 train.py "${args[@]}"