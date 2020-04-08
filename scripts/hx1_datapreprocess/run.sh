#!/bin/bash

data_dir="/homeb/nipeng/data/HX1/BJXWZ-201711003-20X.D1E13.albacore/workspace/pass/"
reference_dir="/homeb/huangneng/NA12878_reference/GRCh38_full_analysis_set_plus_decoy_hla.fa"
fastq_dir="/homeb/huangneng/hx1_fastq"
selected_dir=$fastq_dir"/selected_fast5s"
identity_filter_file="/homeb/huangneng/SACall/hx1_datapreprocess/read_identity_filter.py"

echo data_dir: $data_dir
echo reference_dir: $reference_dir
echo fastq_dir: $fastq_dir
echo selected_dir: $selected_dir


#mkdir $fastq_dir

cd $fastq_dir

mkdir $selected_dir

#nanopolish extract -r $data_dir -o hx1.fasta

minimap2 -x map-ont -c $reference_dir hx1.fasta -t 40 > hx1_align.paf

python $identity_filter_file -i $data_dir -r $reference_dir -q $fastq_dir"/hx1.fasta" -s $fastq_dir"/hx1_align.paf" -o $selected_dir
