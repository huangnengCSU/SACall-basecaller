#!/bin/bash
root_dir=${1}
cd $root_dir
echo `ls $root_dir`
for gdir in `ls $root_dir`
do
    cd $gdir
    echo `pwd`
    minimap2 -ax map-ont reference.fasta sacall.fasta --eqx -t 40 > sacall_aln.sam
    python ~/Projects/sacall/scripts/minimap2_alignment_report.py sacall_aln.sam > sacall.sketch
    python ~/Projects/sacall/scripts/KP_dataset_error_report/read_sketch.py sacall.sketch sacall_error.txt

    minimap2 -ax map-ont reference.fasta albacore.fastq --eqx -t 40 > albacore_aln.sam
    python ~/Projects/sacall/scripts/minimap2_alignment_report.py albacore_aln.sam > albacore.sketch
    python ~/Projects/sacall/scripts/KP_dataset_error_report/read_sketch.py albacore.sketch albacore_error.txt

    minimap2 -ax map-ont reference.fasta guppy213.fastq --eqx -t 40 > guppy213_aln.sam
    python ~/Projects/sacall/scripts/minimap2_alignment_report.py guppy213_aln.sam > guppy213.sketch
    python ~/Projects/sacall/scripts/KP_dataset_error_report/read_sketch.py guppy213.sketch guppy213_error.txt

    minimap2 -ax map-ont reference.fasta guppy213ff.fastq --eqx -t 40 > guppy213ff_aln.sam
    python ~/Projects/sacall/scripts/minimap2_alignment_report.py guppy213ff_aln.sam > guppy213ff.sketch
    python ~/Projects/sacall/scripts/KP_dataset_error_report/read_sketch.py guppy213ff.sketch guppy213ff_error.txt

    minimap2 -ax map-ont reference.fasta guppy213cs.fastq --eqx -t 40 > guppy213cs_aln.sam
    python ~/Projects/sacall/scripts/minimap2_alignment_report.py guppy213cs_aln.sam > guppy213cs.sketch
    python ~/Projects/sacall/scripts/KP_dataset_error_report/read_sketch.py guppy213cs.sketch guppy213cs_error.txt

    minimap2 -ax map-ont reference.fasta guppy237.fastq --eqx -t 40 > guppy237_aln.sam
    python ~/Projects/sacall/scripts/minimap2_alignment_report.py guppy237_aln.sam > guppy237.sketch
    python ~/Projects/sacall/scripts/KP_dataset_error_report/read_sketch.py guppy237.sketch guppy237_error.txt
    cd ../
done
