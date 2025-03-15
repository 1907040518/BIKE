
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2
export CUDA_VISIBLE_DEVICES=5,6
python -m torch.distributed.launch --master_port 1239 --nproc_per_node=2 \
    test_compress.py --config ${config} --weights ${weight} ${@:3}