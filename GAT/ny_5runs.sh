for seed in 10 20 30 40 50;
# for seed in 0;
do
    CUDA_VISIBLE_DEVICES=7 python main.py --n_layer 2 --num_heads 8 --dropout 0.2 --lr 0.05 --seed $seed
done