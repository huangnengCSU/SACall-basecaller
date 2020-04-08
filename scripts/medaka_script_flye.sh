#!/bin/bash

root_dir=$1
cd ${root_dir}
pwd=`pwd`
for dir in `ls ${pwd}`
do
    cd ${pwd}/${dir}
    sacall_reads=sacall.fasta
    albacore_reads=albacore.fastq
    guppy_213=guppy213.fastq
    guppy_213_ff=guppy213ff.fastq
    guppy_213_cs=guppy213cs.fastq
    guppy_237=guppy237.fastq
    reference=reference.fasta
    reference_size=`du -h ${reference} | awk '{print $1}'`
    echo `pwd`
    echo "$reference : $reference_size"
    
    minimap2 -x ava-ont $dir-sacall-flye/assembly.fasta $sacall_reads -t 40 >  $dir-sacall-flye/overlaps_1.paf
    racon $sacall_reads $dir-sacall-flye/overlaps_1.paf $dir-sacall-flye/assembly.fasta -t 40 > $dir-sacall-flye/racon_cons_1.fasta
    minimap2 -x ava-ont $dir-sacall-flye/racon_cons_1.fasta $sacall_reads -t 40 > $dir-sacall-flye/overlaps_2.paf
    racon $sacall_reads $dir-sacall-flye/overlaps_2.paf $dir-sacall-flye/racon_cons_1.fasta -t 40 > $dir-sacall-flye/racon_cons_2.fasta
    minimap2 -x ava-ont $dir-sacall-flye/racon_cons_2.fasta $sacall_reads -t 40 > $dir-sacall-flye/overlaps_3.paf
    racon $sacall_reads $dir-sacall-flye/overlaps_3.paf $dir-sacall-flye/racon_cons_2.fasta -t 40 > $dir-sacall-flye/racon_cons_3.fasta
    minimap2 -x ava-ont $dir-sacall-flye/racon_cons_3.fasta $sacall_reads -t 40 > $dir-sacall-flye/overlaps_4.paf
    racon $sacall_reads $dir-sacall-flye/overlaps_4.paf $dir-sacall-flye/racon_cons_3.fasta -t 40 > $dir-sacall-flye/racon_cons_4.fasta
    medaka_consensus -i $sacall_reads -d $dir-sacall-flye/racon_cons_4.fasta -o $dir-sacall-flye/medaka_polish -t 40 -m r941_trans
    nucmer --prefix="sacall" reference.fasta $dir-sacall-flye/medaka_polish/consensus.fasta
    delta-filter -r -q sacall.delta > sacall.filter
    show-snps -ClrTH -x5 sacall.filter | python ~/Basecalling-comparison/analysis_scripts/error_summary.py "chromosome" "chromosome" >> $dir-sacall-flye/medaka_polish/error_details.tsv
    
    
    minimap2 -x ava-ont $dir-albacore-flye/assembly.fasta $albacore_reads -t 40 >  $dir-albacore-flye/overlaps_1.paf
    racon $albacore_reads $dir-albacore-flye/overlaps_1.paf $dir-albacore-flye/assembly.fasta -t 40 > $dir-albacore-flye/racon_cons_1.fasta
    minimap2 -x ava-ont $dir-albacore-flye/racon_cons_1.fasta $albacore_reads -t 40 > $dir-albacore-flye/overlaps_2.paf
    racon $albacore_reads $dir-albacore-flye/overlaps_2.paf $dir-albacore-flye/racon_cons_1.fasta -t 40 > $dir-albacore-flye/racon_cons_2.fasta
    minimap2 -x ava-ont $dir-albacore-flye/racon_cons_2.fasta $albacore_reads -t 40 > $dir-albacore-flye/overlaps_3.paf
    racon $albacore_reads $dir-albacore-flye/overlaps_3.paf $dir-albacore-flye/racon_cons_2.fasta -t 40 > $dir-albacore-flye/racon_cons_3.fasta
    minimap2 -x ava-ont $dir-albacore-flye/racon_cons_3.fasta $albacore_reads -t 40 > $dir-albacore-flye/overlaps_4.paf
    racon $albacore_reads $dir-albacore-flye/overlaps_4.paf $dir-albacore-flye/racon_cons_3.fasta -t 40 > $dir-albacore-flye/racon_cons_4.fasta
    medaka_consensus -i $albacore_reads -d $dir-albacore-flye/racon_cons_4.fasta -o $dir-albacore-flye/medaka_polish -t 40 -m r941_trans
    nucmer --prefix="albacore" reference.fasta $dir-albacore-flye/medaka_polish/consensus.fasta
    delta-filter -r -q albacore.delta > albacore.filter
    show-snps -ClrTH -x5 albacore.filter | python ~/Basecalling-comparison/analysis_scripts/error_summary.py "chromosome" "chromosome" >> $dir-albacore-flye/medaka_polish/error_details.tsv
    
    
    minimap2 -x ava-ont $dir-guppy213cs-flye/assembly.fasta $guppy_213_cs -t 40 >  $dir-guppy213cs-flye/overlaps_1.paf
    racon $guppy_213_cs $dir-guppy213cs-flye/overlaps_1.paf $dir-guppy213cs-flye/assembly.fasta -t 40 > $dir-guppy213cs-flye/racon_cons_1.fasta
    minimap2 -x ava-ont $dir-guppy213cs-flye/racon_cons_1.fasta $guppy_213_cs -t 40 > $dir-guppy213cs-flye/overlaps_2.paf
    racon $guppy_213_cs $dir-guppy213cs-flye/overlaps_2.paf $dir-guppy213cs-flye/racon_cons_1.fasta -t 40 > $dir-guppy213cs-flye/racon_cons_2.fasta
    minimap2 -x ava-ont $dir-guppy213cs-flye/racon_cons_2.fasta $guppy_213_cs -t 40 > $dir-guppy213cs-flye/overlaps_3.paf
    racon $guppy_213_cs $dir-guppy213cs-flye/overlaps_3.paf $dir-guppy213cs-flye/racon_cons_2.fasta -t 40 > $dir-guppy213cs-flye/racon_cons_3.fasta
    minimap2 -x ava-ont $dir-guppy213cs-flye/racon_cons_3.fasta $guppy_213_cs -t 40 > $dir-guppy213cs-flye/overlaps_4.paf
    racon $guppy_213_cs $dir-guppy213cs-flye/overlaps_4.paf $dir-guppy213cs-flye/racon_cons_3.fasta -t 40 > $dir-guppy213cs-flye/racon_cons_4.fasta
    medaka_consensus -i $guppy_213_cs -d $dir-guppy213cs-flye/racon_cons_4.fasta -o $dir-guppy213cs-flye/medaka_polish -t 40 -m r941_trans
    nucmer --prefix="guppy213cs" reference.fasta $dir-guppy213cs-flye/medaka_polish/consensus.fasta
    delta-filter -r -q guppy213cs.delta > guppy213cs.filter
    show-snps -ClrTH -x5 guppy213cs.filter | python ~/Basecalling-comparison/analysis_scripts/error_summary.py "chromosome" "chromosome" >> $dir-guppy213cs-flye/medaka_polish/error_details.tsv


    minimap2 -x ava-ont $dir-guppy237-flye/assembly.fasta $guppy_237 -t 40 >  $dir-guppy237-flye/overlaps_1.paf
    racon $guppy_237 $dir-guppy237-flye/overlaps_1.paf $dir-guppy237-flye/assembly.fasta -t 40 > $dir-guppy237-flye/racon_cons_1.fasta
    minimap2 -x ava-ont $dir-guppy237-flye/racon_cons_1.fasta $guppy_237 -t 40 > $dir-guppy237-flye/overlaps_2.paf
    racon $guppy_237 $dir-guppy237-flye/overlaps_2.paf $dir-guppy237-flye/racon_cons_1.fasta -t 40 > $dir-guppy237-flye/racon_cons_2.fasta
    minimap2 -x ava-ont $dir-guppy237-flye/racon_cons_2.fasta $guppy_237 -t 40 > $dir-guppy237-flye/overlaps_3.paf
    racon $guppy_237 $dir-guppy237-flye/overlaps_3.paf $dir-guppy237-flye/racon_cons_2.fasta -t 40 > $dir-guppy237-flye/racon_cons_3.fasta
    minimap2 -x ava-ont $dir-guppy237-flye/racon_cons_3.fasta $guppy_237 -t 40 > $dir-guppy237-flye/overlaps_4.paf
    racon $guppy_237 $dir-guppy237-flye/overlaps_4.paf $dir-guppy237-flye/racon_cons_3.fasta -t 40 > $dir-guppy237-flye/racon_cons_4.fasta
    medaka_consensus -i $guppy_237 -d $dir-guppy237-flye/racon_cons_4.fasta -o $dir-guppy237-flye/medaka_polish -t 40 -m r941_trans
    nucmer --prefix="guppy237" reference.fasta $dir-guppy237-flye/medaka_polish/consensus.fasta
    delta-filter -r -q guppy237.delta > guppy237.filter
    show-snps -ClrTH -x5 guppy237.filter | python ~/Basecalling-comparison/analysis_scripts/error_summary.py "chromosome" "chromosome" >> $dir-guppy237-flye/medaka_polish/error_details.tsv

done
