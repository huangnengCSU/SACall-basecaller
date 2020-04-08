#!/bin/bash

root_dir=$1
scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${root_dir}
pwd=`pwd`
for dir in `ls ${pwd}`
do
    cd ${pwd}/${dir}
    echo `pwd`

    cd $dir-sacall-flye
    python $scripts_dir/chop_up_assembly.py assembly.fasta 10000 > assembly_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta assembly_piece.fasta > assembly_piece.paf
    python $scripts_dir/read_length_identity.py assembly_piece.fasta assembly_piece.paf > assembly_identity.tsv
    cd ..
    
    cd $dir-albacore-flye
    python $scripts_dir/chop_up_assembly.py assembly.fasta 10000 > assembly_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta assembly_piece.fasta > assembly_piece.paf
    python $scripts_dir/read_length_identity.py assembly_piece.fasta assembly_piece.paf > assembly_identity.tsv
    cd ..

    cd $dir-guppy213cs-flye
    python $scripts_dir/chop_up_assembly.py assembly.fasta 10000 > assembly_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta assembly_piece.fasta > assembly_piece.paf
    python $scripts_dir/read_length_identity.py assembly_piece.fasta assembly_piece.paf > assembly_identity.tsv
    cd ..

    cd $dir-guppy237-flye
    python $scripts_dir/chop_up_assembly.py assembly.fasta 10000 > assembly_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta assembly_piece.fasta > assembly_piece.paf
    python $scripts_dir/read_length_identity.py assembly_piece.fasta assembly_piece.paf > assembly_identity.tsv
    cd ..

done
