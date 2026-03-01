model_name=CPTransformer
dataset=NAion
train_epochs=100
early_cycle_threshold=100
learning_rate=0.00001
master_port=25215
num_process=0
batch_size=64
n_heads=2
seq_len=10
accumulation_steps=4
e_layers=7
d_layers=4 # todo: try d=4, o=1 vs d=1, o=4
d_model=32
d_ff=64
dropout=0.09481290915677548
charge_discharge_length=300
patience=15 # Eearly stopping patience
lradj=COS
loss=MSE

checkpoints=/out/checkpoints/cpt # the save path of checkpoints
data=Dataset_original
root_path=./dataset
comment='CPTransformer'
task_name=classification

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name $task_name \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id CPTransformer \
  --model $model_name \
  --features MS \
  --seq_len $seq_len \
  --label_len 50 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --dataset $dataset \
  --num_workers 1 \
  --e_layers $e_layers \
  --lstm_layers 6 \
  --d_layers $d_layers \
  --patience $patience \
  --n_heads $n_heads \
  --early_cycle_threshold $early_cycle_threshold \
  --dropout $dropout \
  --lradj $lradj \
  --loss $loss \
  --checkpoints $checkpoints \
  --wd 0.0006710947780627601

