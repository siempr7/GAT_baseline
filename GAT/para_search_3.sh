
# for n_layer in 1 2;
# do
#     for lr in 0.005;
#     do
#         for dropout in 0 0.2;
#         do
#             for num_heads in 4 8;
#             do
#                 CUDA_VISIBLE_DEVICES=4 python main.py --lr $lr  --dropout $dropout --n_layer $n_layer --num_heads $num_heads
#             done
#         done
#     done
# done