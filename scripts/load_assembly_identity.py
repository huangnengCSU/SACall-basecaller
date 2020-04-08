import pandas as pd
import sys
import os

data_dir = sys.argv[1]
base_name = os.path.basename(data_dir)

sacall_identity_path=data_dir+'/'+base_name+'-sacall-flye'
albacore_identity_path=data_dir+'/'+base_name+'-albacore-flye'
guppy_KP_identity_path=data_dir+'/'+base_name+'-guppy213cs-flye'
guppy_identity_path=data_dir+'/'+base_name+'-guppy237-flye'

df = pd.read_csv(sacall_identity_path+'/'+'assembly_identity.tsv',sep='\t')
print('sacall:',df['Identity'].median())
df = pd.read_csv(guppy_KP_identity_path+'/'+'assembly_identity.tsv',sep='\t')
print('guppy-kp:',df['Identity'].median())
df = pd.read_csv(guppy_identity_path+'/'+'assembly_identity.tsv',sep='\t')
print('guppy:',df['Identity'].median())
df = pd.read_csv(albacore_identity_path+'/'+'assembly_identity.tsv',sep='\t')
print('albacore:',df['Identity'].median())
