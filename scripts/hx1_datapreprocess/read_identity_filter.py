import os
import argparse
import subprocess
import h5py
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path to source fast5 data', required=True)
parser.add_argument('-r', help='path to reference', required=True)
parser.add_argument('-q', help='path to basecalled fastq file', required=True)
parser.add_argument('-s', help='path to sam file', required=True)
parser.add_argument('-o', help='path to selected reads', required=True)
argv = parser.parse_args()

path_to_source_fast5 = argv.i
path_to_reference = argv.r
path_to_selected = argv.o
path_to_fastq = argv.q
path_to_sam = argv.s

if not os.path.exists(path_to_selected):
    os.mkdir(path_to_selected)

# get the mapping of fast5 file absolute path and fast5 read id
for dirs in os.listdir(path_to_source_fast5):
    sub_path = os.path.join(path_to_source_fast5,dirs)
    if not os.path.isdir(sub_path):
        continue
    path_rid_map = {}
    for f5 in os.listdir(sub_path):
        if not f5.endswith('.fast5'):
            continue
        f5read = h5py.File(sub_path+'/'+f5,'r')
        read = f5read['Raw/Reads']
        read = read[list(read.keys())[0]]
        read_id = read.attrs['read_id'].decode('UTF-8')
        path_rid_map[read_id] = sub_path+'/'+f5
with open('path_rid_map.json','w') as outfile:
    json.dump(path_rid_map,outfile,ensure_ascii=False)
    outfile.write('\n')
identity_file = 'call.identity'
read_identity_cmd = 'python read_length_identity.py {fastq} {sam} > {identity}'.format(fastq=path_to_fastq,sam=path_to_sam,identity=identity_file)
f = subprocess.run(read_identity_cmd,shell=True)
iden = pd.read_csv(identity_file, header=0)
select = iden[(iden['Identity'] > 90) & (
    iden['Length'] > 10000)]['Name'].to_list()
select_ids = [v.split('_')[0] for v in select]

for rid in select_ids:
    cp_cmd = 'cp {select_read} {dst}'.format(select_read=path_rid_map[rid], dst=path_to_selected)
    f = subprocess.run(cp_cmd, shell=True)

