args_path=/path/to/your/saving/folder/ # the model you want to evaluate
batch_size=16
num_process=2
master_port=26949
eval_cycle_min=1 # set eval_cycle_min as 1 and eval_cycle_max as 100 to evaluate all samples
eval_cycle_max=100
eval_dataset=CALB
model=CPMLP

CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --multi_gpu --num_processes $num_process --main_process_port $master_port evaluate_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --model $model