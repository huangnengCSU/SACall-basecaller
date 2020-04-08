#!/bin/bash


model_path=$1
fast5_dir=$2
signal_window_length=$3
basecalled_filename=$4
tmp_records_dir="tmp_data_dir"

# data preprocessing
python generate_dataset/inference_data.py -fast5 ${fast5_dir} -records_dir ${tmp_records_dir} -raw_len ${signal_window_length}

# caller
python call.py -model ${model_path} -records_dir ${tmp_records_dir} -output ${basecalled_filename}

# delete tmp records
rm -rf ${tmp_records_dir}





