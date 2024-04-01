SESSION='avss'
CONFIG="config/avss/AVSegFormer_pvt2_avss.py"
WEIGHTS='work_dir/AVSegFormer_pvt2_avss/AVSS_miou_best.pth'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/$SESSION/test.py \
        $CONFIG \
        $WEIGHTS \
        --save_pred_mask
