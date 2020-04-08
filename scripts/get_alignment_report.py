import sys
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


def get_alignment_report(read_length_identity_dir):
    """
        remove duplicates of read alignment by reserving the longest
    """
    df = pd.read_csv(read_length_identity_dir, header=0, sep='\t')

    # drop Nan
    df.dropna(axis=0, how='any', inplace=True)

    # sort by "Name" and "Length"
    df.sort_values(by=['Name', 'Length'], ascending=False, inplace=True)

    # drop_duplicates by "Name"
    df.drop_duplicates(['Name'], keep='first', inplace=True)

    return df


def draw_distribution(dataframe, fig_name):
    fig = plt.figure(figsize=(16,9))
    reads_identity = dataframe['Identity']
    hist, bin_edges = np.histogram(
        reads_identity, bins=np.arange(50, 100, 0.5))
    plt.hist(reads_identity, bins=bin_edges, facecolor='g', alpha=0.2)
    plt.xlabel('read identity')
    plt.ylabel('count')
    plt.title(os.path.splitext(os.path.basename(fig_name))[0].replace('_',' '))
    plt.savefig(fig_name)


if __name__ == "__main__":
    file_dir = sys.argv[1]
    df = get_alignment_report(file_dir)
    fig_name = os.path.splitext(file_dir)[0]+'_identity_distribution.png'
    draw_distribution(df, fig_name)
