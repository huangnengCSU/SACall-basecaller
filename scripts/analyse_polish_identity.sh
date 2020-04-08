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
    # python $scripts_dir/chop_up_assembly.py assembly.fasta 10000 > assembly_piece.fasta
    # minimap2 -x asm5 -t 20 -c ../reference.fasta assembly_piece.fasta > assembly_piece.paf
    # python $scripts_dir/read_length_identity.py assembly_piece.fasta assembly_piece.paf > assembly_identity.tsv

    for i in {1..4}
    do
    echo `pwd`
    python $scripts_dir/chop_up_assembly.py racon_cons_${i}.fasta 10000 > racon_cons_${i}_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta racon_cons_${i}_piece.fasta > racon_cons_${i}_piece.paf
    python $scripts_dir/read_length_identity.py racon_cons_${i}_piece.fasta racon_cons_${i}_piece.paf > racon_cons_${i}_identity.tsv
    done

    python $scripts_dir/chop_up_assembly.py medaka_polish/consensus.fasta 10000 > medaka_consensus_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta medaka_consensus_piece.fasta > medaka_consensus_piece.paf
    python $scripts_dir/read_length_identity.py medaka_consensus_piece.fasta medaka_consensus_piece.paf > medaka_consensus_identity.tsv
    cd ..
    

    cd $dir-albacore-flye
    # python $scripts_dir/chop_up_assembly.py assembly.fasta 10000 > assembly_piece.fasta
    # minimap2 -x asm5 -t 20 -c ../reference.fasta assembly_piece.fasta > assembly_piece.paf
    # python $scripts_dir/read_length_identity.py assembly_piece.fasta assembly_piece.paf > assembly_identity.tsv

    for i in {1..4}
    do
    echo `pwd`
    python $scripts_dir/chop_up_assembly.py racon_cons_${i}.fasta 10000 > racon_cons_${i}_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta racon_cons_${i}_piece.fasta > racon_cons_${i}_piece.paf
    python $scripts_dir/read_length_identity.py racon_cons_${i}_piece.fasta racon_cons_${i}_piece.paf > racon_cons_${i}_identity.tsv
    done

    python $scripts_dir/chop_up_assembly.py medaka_polish/consensus.fasta 10000 > medaka_consensus_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta medaka_consensus_piece.fasta > medaka_consensus_piece.paf
    python $scripts_dir/read_length_identity.py medaka_consensus_piece.fasta medaka_consensus_piece.paf > medaka_consensus_identity.tsv
    cd ..

    cd $dir-guppy213cs-flye
    # python $scripts_dir/chop_up_assembly.py assembly.fasta 10000 > assembly_piece.fasta
    # minimap2 -x asm5 -t 20 -c ../reference.fasta assembly_piece.fasta > assembly_piece.paf
    # python $scripts_dir/read_length_identity.py assembly_piece.fasta assembly_piece.paf > assembly_identity.tsv

    for i in {1..4}
    do
    echo `pwd`
    python $scripts_dir/chop_up_assembly.py racon_cons_${i}.fasta 10000 > racon_cons_${i}_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta racon_cons_${i}_piece.fasta > racon_cons_${i}_piece.paf
    python $scripts_dir/read_length_identity.py racon_cons_${i}_piece.fasta racon_cons_${i}_piece.paf > racon_cons_${i}_identity.tsv
    done

    python $scripts_dir/chop_up_assembly.py medaka_polish/consensus.fasta 10000 > medaka_consensus_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta medaka_consensus_piece.fasta > medaka_consensus_piece.paf
    python $scripts_dir/read_length_identity.py medaka_consensus_piece.fasta medaka_consensus_piece.paf > medaka_consensus_identity.tsv
    cd ..

    cd $dir-guppy237-flye
    # python $scripts_dir/chop_up_assembly.py assembly.fasta 10000 > assembly_piece.fasta
    # minimap2 -x asm5 -t 20 -c ../reference.fasta assembly_piece.fasta > assembly_piece.paf
    # python $scripts_dir/read_length_identity.py assembly_piece.fasta assembly_piece.paf > assembly_identity.tsv

    for i in {1..4}
    do
    echo `pwd`
    python $scripts_dir/chop_up_assembly.py racon_cons_${i}.fasta 10000 > racon_cons_${i}_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta racon_cons_${i}_piece.fasta > racon_cons_${i}_piece.paf
    python $scripts_dir/read_length_identity.py racon_cons_${i}_piece.fasta racon_cons_${i}_piece.paf > racon_cons_${i}_identity.tsv
    done

    python $scripts_dir/chop_up_assembly.py medaka_polish/consensus.fasta 10000 > medaka_consensus_piece.fasta
    minimap2 -x asm5 -t 20 -c ../reference.fasta medaka_consensus_piece.fasta > medaka_consensus_piece.paf
    python $scripts_dir/read_length_identity.py medaka_consensus_piece.fasta medaka_consensus_piece.paf > medaka_consensus_identity.tsv
    cd ..

done
