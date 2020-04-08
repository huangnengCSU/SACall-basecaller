#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename: filter_reads
# @Date    : 04/16/2019
# @Author  : Neng Huang
# @Email   : csuhuangneng@gmail.com

import sys
import os
import shutil
import h5py
import re
import pandas as pd

fast5_dir = sys.argv[1]
sam_file = sys.argv[2]
output_dir = sys.argv[3]
min_len = int(sys.argv[4])

fields = ['QNAME', 'FLAG', 'RNAME', 'POS', 'MAPQ', 'CIGAR', 'RNEXT', 'PNEXT', 'TLEN', 'SEQ', 'QUAL']
cigar_patten = '([0-9]+[MIDNSHPX=])'
num_flag_cigar = '[0-9]+|[MIDNSHPX=]'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
else:
    os.mkdir(output_dir)

# get read name id mapping
for dirs in os.listdir(fast5_dir):
    sub_path = os.path.join(fast5_dir, dirs)
    if not os.path.isdir(sub_path):
        continue
    path_rid_map = {}
    for f5 in os.listdir(sub_path):
        if not f5.endswith('.fast5'):
            continue
        f5read = h5py.File(sub_path + '/' + f5, 'r')
        read = f5read['Raw/Reads']
        read = read[list(read.keys())[0]]
        read_id = read.attrs['read_id'].decode('UTF-8')
        path_rid_map[read_id] = sub_path + '/' + f5

aln_records = list()
with open(sam_file, 'r') as fsam:
    for line in fsam:
        if line.startswith('@'):
            continue
        sections = dict(zip(fields, line.strip().split('\t')))
        ref_len, query_len, map_len, err_num = 0, 0, 0, 0
        win_size, win_ins, win_del = 0, 0, 0

        if sections['CIGAR'] == '*':
            continue
        for match in re.finditer(cigar_patten, sections['CIGAR']):
            melem = match.group()
            opt_num, opt_flag = re.findall(num_flag_cigar, melem)
            opt_num = int(opt_num)
            if opt_flag == 'X':
                err_num += opt_num
                ref_len += opt_num
                query_len += opt_num
                win_size += opt_num
            elif opt_flag == '=':
                map_len += opt_num
                ref_len += opt_num
                query_len += opt_num
                win_size += opt_num
            elif opt_flag == 'D':
                if opt_num > 15:
                    break
                err_num += opt_num
                ref_len += opt_num
                win_size += opt_num
                win_del += opt_num
            elif opt_flag == 'I':
                if opt_num > 15:
                    break
                err_num += opt_num
                query_len += opt_num
                win_size += opt_num
                win_ins += opt_num
            if win_size > 50:
                if (win_ins + win_del) / win_size > 0.8:
                    break
                win_size, win_ins, win_del = 0, 0, 0
        # mapping identity, error rate, mapping len
        aln_records.append([sections['QNAME'], map_len / ref_len, err_num / ref_len, query_len])

df = pd.DataFrame(aln_records, columns=['name', 'identity', 'error', 'length'])
df = df[df['length'] >= min_len]
df.sort_values(by=['name', 'identity'], ascending=False, inplace=True)
df.drop_duplicates(['name'], keep='first', inplace=True)
names = df[(df['identity'] >= 0.92) & (df['error'] <= 0.08)]['name'].tolist()
print(len(names))

for name in names:
    try:
        cp_cmd = "cp {src} {dst}".format(src=path_rid_map[name], dst=output_dir)
        os.system(cp_cmd)
    except:
        print('fast5 file dont exist')
