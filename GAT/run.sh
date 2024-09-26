for i in 1;
do
    CUDA_VISIBLE_DEVICES=0 python main.py --num_iterations 5 --lr 0.005 --n_layer 1
done
