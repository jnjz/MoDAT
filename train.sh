SESSION='avss'
CONFIG="config/avss/AVSegFormer_pvt2_avss.py"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/$SESSION/train.py $CONFIG
