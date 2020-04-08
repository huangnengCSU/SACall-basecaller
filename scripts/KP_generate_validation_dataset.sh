#!/bin/bash
basecall_compare_data_root=$1
merge_kp_path=$2
signal_out_path=$3
label_out_path=$4

mkdir ${merge_kp_path}
mkdir ${signal_out_path}
mkdir ${label_out_path}

for p in `ls ${basecall_compare_data_root}`
do
    datapath="${basecall_compare_data_root}/${p}"
    refpath="${datapath}/read_references.fasta"
    fast5path="${datapath}/validation_fast5s"
    basecalledfast5path="${datapath}/validation_fast5s_guppy"

    # guppy
    printf "\n"
    printf "guppy caller running..."
    guppy_basecaller --kit SQK-LSK108 --flowcell FLO-MIN106 --fast5_out --input ${fast5path} --save_path ${basecalledfast5path} --num_callers 40


    # tombo
    printf "\n"
    printf "tombo running ..."
    tombo resquiggle ${basecalledfast5path}/workspace/ ${refpath} --overwrite --processes 40

    # copy
    printf "\n"
    echo "copy fast5s to ${merge_kp_path}"
    cp ${basecalledfast5path}/workspace/* ${merge_kp_path}
done

printf "\n"
echo "generate signal and label into ${signal_out_path} and ${label_out_path}..."
python ../generate_dataset/train_data.py -i ${merge_kp_path} -so ${signal_out_path} -lo ${label_out_path} -raw_len 2048 -seq_len 512 -t 40
echo "done"


