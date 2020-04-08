import sys
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
data_names = ['14260550', '14260541', '14260517', '14260556',
           '14260514', '14260574', '14260562', '14260511', '14260568']
genome_name = ['Klebsiella Pneumoniae NUH29', 'Klebsiella Pneumoniae KSB2', 'Klebsiella Pneumoniae INF042', 'Serratia Marcescens',
               'Haemophilus Haemolyticus', 'Stenotrophomonas Maltophilia', 'Shigella Sonnei', 'Acinetobacter Pittii', 'Staphylococcus Aureus']
basecaller_lst = ['SACall', 'Guppy-KP', 'Guppy', 'Albacore']
validate_data_dir = sys.argv[1]

df_lst = []
for data_id in data_names:
    sacall_flye_assembly_dir = data_id + "-sacall-flye"
    guppy_kp_flye_assembly_dir = data_id + "-guppy213cs-flye"
    guppy_assembly_dir = data_id + "-guppy237-flye"
    albacore_assembly_dir = data_id + "-albacore-flye"
    identity_dict = dict()
    for idx, basecaller_dir in enumerate([
            sacall_flye_assembly_dir,
            guppy_kp_flye_assembly_dir,
            guppy_assembly_dir,
            albacore_assembly_dir,
    ]):
        idientity_lst = []
        for i in range(1, 5):
            tmp_df = pd.read_csv(
                validate_data_dir + "/" + data_id + "/" + basecaller_dir +
                "/" + f"racon_cons_{i:d}_identity.tsv",
                sep="\t",
            )
            identity = tmp_df["Identity"].median()
            idientity_lst.append(identity)
        tmp_df = pd.read_csv(
            validate_data_dir + "/" + data_id + "/" + basecaller_dir + "/" +
            "medaka_consensus_identity.tsv",
            sep="\t",
        )
        identity = tmp_df["Identity"].median()
        idientity_lst.append(identity)
        # print(data_id, basecaller_dir, idientity_lst)
        identity_dict[basecaller_lst[idx]] = idientity_lst
    df_lst.append(pd.DataFrame(identity_dict))

fig = plt.figure(figsize=(16,16))
for i in range(1, 10):
    ax = fig.add_subplot(3, 3, i)
    df = df_lst[i-1]
    ax.plot(df.index, df['SACall'], linewidth=2, marker='s')
    ax.plot(df.index, df['Guppy-KP'], linewidth=2, marker='>')
    ax.plot(df.index, df['Guppy'], linewidth=2, marker='o')
    ax.plot(df.index, df['Albacore'], linewidth=2, marker='*')
    ax.set_title(genome_name[i-1])
    if i==3:
        plt.legend(basecaller_lst)
    plt.grid()
    plt.xlim((0, 4))
    plt.ylim((99,100))
    x_ticks = np.arange(0,5)
    plt.xticks(x_ticks)
    y_ticks = np.arange(99,100,0.2)
    plt.yticks(y_ticks)
plt.savefig('polish_identity.png')
