#!/bin/bash

basecall_name=$1
reference="reference.fasta"
threads=20

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
filename=$(echo ${basecall_name} | sed 's/\.[^.]*$//')
result_dir="${filename}"_result
read_alignment="${result_dir}"/"${filename}"_reads.paf
read_data="${result_dir}"/"${filename}"_reads.tsv


echo "reads alignment: minimap2..."
printf "\n"

mkdir -p ${result_dir}
minimap2 -x map-ont -t ${threads} -c ${reference} ${basecall_name} > ${read_alignment}
pypy3 "${scripts_dir}"/read_length_identity.py ${basecall_name} ${read_alignment} > ${read_data}

echo "draw reads identity distribution..."
printf "\n"
python "${scripts_dir}"/get_alignment_report.py ${read_data}


