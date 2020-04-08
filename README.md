#SACall
SACall: a neural network basecaller for Oxford Nanopore sequencing data based on self-attention mechanism.

####DNA basecalling command
```angular2
bash run_caller.sh <model file> <fast5 folder> <signal window length> <output file name>
```
command parameters  
`model file`: we provide the `model.chkpt` trained on Klebsiella pneumoniae genome.   
`fast5 folder`: directory of original sequencing files.   
`signal window length`: the length of the signal segment, default: `2048`.   
`output file name`: the name of basecalled file.  

####Installation
* python3  
* pytorch v1.0.1    
* ctcdecode: https://github.com/parlance/ctcdecode.git
