#!/bin/bash

. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

AIR_dataset_dir=$1
AIR_recipe_dir=`pwd`

data_dir=data/local/data

python3 local/kaldi_file_preparation.py $AIR_dataset_dir $AIR_recipe_dir $data_dir

echo "Data preparation done successfully!!!"
