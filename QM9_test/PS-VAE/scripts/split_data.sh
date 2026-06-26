#!/bin/bash
# Created Time: 2026-03-05
# Description: Split dataset into train/valid/test using split_data.py
#########################################################################

CODE_DIR=`dirname $0`/../src
DATA_DIR=`dirname $0`/../data

# 如果 split_data.py 不在 scripts 目录，把下面路径改成实际位置
SPLIT_PY=${CODE_DIR}/data/split.py

# 输入数据（原始 txt 文件，逐行一个 SMILES）
IN_DATA=${DATA_DIR}/my_zinc/my_zinc250k.txt

# 输出目录（会自动创建 train/ valid/ test 子目录）
OUT_DIR=${DATA_DIR}/my_zinc/

# 划分比例
VALID_RATIO=0.10
TEST_RATIO=0.10
SEED=6

echo "CODE_DIR: ${CODE_DIR}"
echo "DATA_DIR: ${DATA_DIR}"
echo "SPLIT_PY: ${SPLIT_PY}"
echo "IN_DATA:  ${IN_DATA}"
echo "OUT_DIR:  ${OUT_DIR}"
echo "VALID_RATIO: ${VALID_RATIO}, TEST_RATIO: ${TEST_RATIO}, SEED: ${SEED}"

python ${SPLIT_PY} \
    --data ${IN_DATA} \
    --valid_ratio ${VALID_RATIO} \
    --test_ratio ${TEST_RATIO} \
    --output_dir ${OUT_DIR} \
    --seed ${SEED}