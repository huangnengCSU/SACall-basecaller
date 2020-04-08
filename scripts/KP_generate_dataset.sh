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
    fast5path="${datapath}/training_fast5s/0000"
    trimed_fast5s_02="${datapath}/sacall_trimed_fast5s_02"
    basecalling_03="${datapath}/sacall_basecalling_03"
    trimed_fast5s_04="${datapath}/sacall_trimed_fast5s_04"
    basecalledfast5path="${datapath}/training_fast5s_guppy"

    # signal-level triming of fast5s
    python rrwick_codes/trim_signal.py --trim_amount 2000 --min_size 50000 ${fast5path} ${trimed_fast5s_02}

    # guppy caller
    guppy_basecaller --kit SQK-LSK108 --flowcell FLO-MIN106 --fast5_out --input ${trimed_fast5s_02} --save_path ${basecalling_03} --num_callers 40

    # minimap2 alignment
    cat ${basecalling_03}/*.fastq > ${basecalling_03}/temp.fastq
    minimap2 -c -x map-ont -t 20 ${refpath} ${basecalling_03}/temp.fastq > ${basecalling_03}/alignments.paf
    rm ${basecalling_03}/temp.fastq

    # Alignment-based QC filter
    python rrwick_codes/filter_reads.py --min_basecalled_length 5000 --max_unaligned_bases 30 --max_window_indels 0.8 --window_size 25 ${trimed_fast5s_02} ${basecalling_03}/sequencing_summary.txt ${refpath} ${basecalling_03}/alignments.paf ${trimed_fast5s_04} ${basecalling_03}/read_references.fasta
    rm ${basecalling_03}/read_references.fasta

    # guppy
    printf "\n"
    printf "guppy caller running..."
    guppy_basecaller --kit SQK-LSK108 --flowcell FLO-MIN106 --fast5_out --input ${trimed_fast5s_04} --save_path ${basecalledfast5path} --num_callers 40

    # delete trimed_fast5s_02,basecalling_03,trimed_fast5s_04
    rm -rf ${trimed_fast5s_02}
    rm -rf ${basecalling_03}
    rm -rf ${trimed_fast5s_04}


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


