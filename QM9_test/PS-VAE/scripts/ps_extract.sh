#!/bin/bash
##########################################################################
# File Name: ps_extract.sh
# Author: kxz
# mail: jackie_kxz@outlook.com
# Created Time: Monday, September 26, 2022 PM02:29:25 HKT
#########################################################################



CODE_DIR=`dirname $0`/../src
DATA_DIR=`dirname $0`/../data
CKPT_DIR=`dirname $0`/../ckpts

export PYTHONPATH=$CODE_DIR:$PYTHONPATH
export OMP_NUM_THREADS=2

python ${CODE_DIR}/data/mol_bpe.py \
    --data ${DATA_DIR}/my_zinc/train/train.txt \
    --output ${DATA_DIR}/vocab/vocab_zinc_bpe_1000.txt \
    --vocab_size 1000 \
    --workers 2
