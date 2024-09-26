
for n_layer in 2;
do
    # for lr in 0.0005 0.001 0.005;
    for lr in 0.01 0.05;
    do
        for dropout in 0.2;
        do
            for num_heads in 8;
            do
                CUDA_VISIBLE_DEVICES=4 python main.py --lr $lr  --dropout $dropout --n_layer $n_layer --num_heads $num_heads
            done
        done
    done
done