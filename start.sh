[ -z "${exp_name}" ] && exp_name="cora"
[ -z "${epoch}" ] && epoch="1000"
[ -z "${seed}" ] && seed="2022"
[ -z "${arch}" ] && arch="--ffn_dim 128 --hidden_dim 128 --dropout_rate 0.1 --n_layers 4 --peak_lr 2e-4"
[ -z "${batch_size}" ] && batch_size="32"
[ -z "${data_augment}" ] && data_augment="8"

max_epochs=$((epoch+1))
echo "=====================================ARGS======================================"
echo "max_epochs: ${max_epochs}"
echo "==============================================================================="


echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "==============================================================================="

default_root_dir="./exps/$exp_name/$seed"
mkdir -p $default_root_dir

python main_adaptive.py --seed $seed --batch_size $batch_size \
      --dataset_name $exp_name --epochs $epoch\
      $arch \
      --checkpoint_path $default_root_dir\
      --num_data_augment $data_augment


