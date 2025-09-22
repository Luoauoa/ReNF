export CUDA_VISIBLE_DEVICES=7

if [ ! -d "./logs_" ]; then
    mkdir ./logs_
fi

if [ ! -d "./logs_/LongForecasting02" ]; then
    mkdir ./logs_/LongForecasting02
fi
seq_len=512
model_name=GoodModel

root_path_name=../dataset/traffic/
data_path_name=(traffic.csv)
model_id_name=(traffic)
data_name=(custom)

random_seed=2021
# traffic
# for pred_len in 96 192
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[0]} \
#       --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[0]} \
#       --features M \
#       --jepa 0 \
#       --revin 1\
#       --d_layers 4\
#       --n_block 2\
#       --pe 0\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.999\
#       --d_ff 2048\
#       --alpha_freq 0.2\
#       --dropout 0.3\
#       --des 'Exp' \
#       --train_epochs 45\
#       --patience 15 \
#       --gamma 0.8\
#       --itr 1 --batch_size 64 --learning_rate 1e-3 >logs_/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done


# for pred_len in 336
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[0]} \
#       --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[0]} \
#       --features M \
#       --jepa 0 \
#       --revin 1\
#       --d_layers 4\
#       --n_block 2\
#       --pe 0\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.998\
#       --d_ff 2048\
#       --alpha_freq 0.2\
#       --dropout 0.3\
#       --des 'Exp' \
#       --train_epochs 45\
#       --patience 15 \
#       --gamma 0.8\
#       --itr 1 --batch_size 64 --learning_rate 1e-3 >logs_/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done


for pred_len in 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name[0]} \
      --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name[0]} \
      --features M \
      --jepa 0 \
      --revin 1\
      --d_layers 6\
      --n_block 1\
      --pe 0\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --d_ff 2048\
      --alpha_freq 0.2\
      --dropout 0.5\
      --des 'Exp' \
      --train_epochs 35\
      --patience 15 \
      --gamma 0.9\
      --itr 1 --batch_size 64 --learning_rate 8e-4 >logs_/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done