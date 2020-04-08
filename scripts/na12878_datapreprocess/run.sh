#!/usr/bin/env bash


called_fastq=$1
genome=$2
fast5=$3
output=$4
min_len=$5


minimap2 -ax map-ont ${genome} ${called_fastq} -t 40 --eqx > aln.sam

python filter_reads.py ${fast5} aln.sam ${output} ${min_len}

rm aln.sam
