'''
 * @Author: huangneng 
 * @Date: 2019-05-28 19:19:32 
 * @Last Modified by:   huangneng 
 * @Last Modified time: 2019-05-28 19:19:32 
'''

import sys
sys.path.append('..')
import constants

import numpy as np
import argparse
import os
import h5py
from statsmodels import robust
import random
from itertools import repeat
from multiprocessing import Pool
import shutil


def splitdata(signal,
              starts,
              ends,
              bases,
              raw_win_size,
              seq_win_size):
    signal_list = []
    base_list = []

    def init():
        raw = []
        seq = []
        raw_len = 0
        return raw, seq, raw_len

    # normalization
    signal = (signal - np.median(signal)) / np.float(robust.mad(signal))

    raw, seq, raw_len = init()
    for idx, s in enumerate(starts):
        e = ends[idx]
        if e - s > raw_win_size:
            # one base maps too more signals
            raw, seq, raw_len = init()
            continue
        elif e - s > 128 or e - s <= 1:
            # one base maps too many signals or too few signals
            raw, seq, raw_len = init()
            continue
        else:
            if raw_len + (e - s) <= raw_win_size:
                raw.extend(signal[s:e])
                seq.append(constants.BASE_DIC.get(bases[idx]))
                raw_len += (e - s)
            else:
                if seq_win_size - len(seq) < 0:
                    # seq length greater than seq_max_len
                    raw, seq, raw_len = init()
                    continue
                # elif len(seq) > raw_win_size // 6:
                #     # signal sample rate too small
                #     raw, seq, raw_len = init()
                #     continue
                elif len(raw) < raw_win_size // 2:
                    # too small signals
                    raw, seq, raw_len = init()
                    continue
                else:
                    if raw == [] or seq == []:
                        raw, seq, raw_len = init()
                        continue
                    else:
                        signal_list.append(np.pad(
                            raw, (0, raw_win_size-len(raw)), mode='constant', constant_values=constants.SIG_PAD))
                        base_list.append(np.pad(
                            seq, (0, seq_win_size-len(seq)), mode='constant', constant_values=constants.PAD))
                        raw, seq, raw_len = init()
    return signal_list, base_list


def get_label_raw(fast5_fn, basecall_group, basecall_subgroup, reverse=False):
    # Open file
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get raw data
    try:
        read = fast5_data['Raw/Reads']
        read_id = read[list(read.keys())[0]].attrs['read_id'].decode('UTF-8')

        raw_attr = fast5_data['Raw/Reads/']
        read_name = list(raw_attr.keys())[0]
        raw_dat = raw_attr[read_name + '/Signal'][()]
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
            'new segments cannot be identified.')

    # Read corrected data
    try:
        corr_data = fast5_data[
            '/Analyses/'+basecall_group + '/' + basecall_subgroup + '/Events']
        corr_attrs = dict(list(corr_data.attrs.items()))
        corr_data = corr_data[()]
    except:
        raise RuntimeError((
            'Corrected data not found.'))

    fast5_info = fast5_data['UniqueGlobalKey/channel_id'].attrs
    # sampling_rate = fast5_info['sampling_rate'].astype('int_')

    # Reading extra information
    corr_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']  #
    if len(raw_dat) > 99999999:
        raise ValueError(fast5_fn + ": max signal length exceed 99999999")
    if any(len(vals) <= 1 for vals in (
            corr_data, raw_dat)):
        raise NotImplementedError((
            'One or no segments or signal present in read.'))
    event_starts = corr_data['start'] + corr_start_rel_to_raw
    event_lengths = corr_data['length']
    event_bases = corr_data['base']

    fast5_data.close()
    return (raw_dat, event_bases, event_starts, event_lengths, read_id)


def extract_file(input_file, raw_len, seq_len, signal_output_dir, label_output_dir, basecall_group, basecall_subgroup):
    try:
        (raw_data, raw_base, raw_start, raw_length, read_id) = get_label_raw(
            input_file, basecall_group, basecall_subgroup)
    except:
        return 0

    starts = []
    ends = []
    bases = []
    for index, start in enumerate(raw_start):
        if raw_length[index] == 0:
            print("input_file:" + input_file)
            raise ValueError("catch a label with length 0")
        starts.append(start)
        ends.append(start + raw_length[index])
        bases.append(bytes.decode(raw_base[index]))
    signal_list, base_list = splitdata(
        raw_data, starts, ends, bases, raw_len, seq_len)
    if signal_list != [] and base_list != []:
        np.save(signal_output_dir+'/'+read_id, np.array(signal_list))
        np.save(label_output_dir+'/'+read_id, np.array(base_list))
        return 1
    else:
        return 0


def generate_train_data(input_dir, raw_len, seq_len, signal_output_dir, label_output_dir, basecall_group='RawGenomeCorrected_000', basecall_subgroup='BaseCalled_template', n_thread=20):
    if not os.path.exists(input_dir):
        print('input dir not exist.')
        os._exit()
    if not os.path.exists(signal_output_dir):
        os.mkdir(signal_output_dir)
    else:
        shutil.rmtree(signal_output_dir)
        os.mkdir(signal_output_dir)
    if not os.path.exists(label_output_dir):
        os.mkdir(label_output_dir)
    else:
        shutil.rmtree(label_output_dir)
        os.mkdir(label_output_dir)
    rawfile_list = [input_dir+'/'+v for v in os.listdir(input_dir)]
    extract_args = zip(rawfile_list, repeat(raw_len), repeat(seq_len), repeat(
        signal_output_dir), repeat(label_output_dir), repeat(basecall_group), repeat(basecall_subgroup))
    pool = Pool(n_thread)
    R = pool.starmap(extract_file, extract_args)
    success_count = sum(list(R))
    total_count = len(R)
    print('success number:', success_count)
    print('fail number:', total_count-success_count)


def run(args):
    global FLAGS
    FLAGS = args
    generate_train_data(input_dir=FLAGS.input,
                        raw_len=FLAGS.raw_len,
                        seq_len=FLAGS.seq_len,
                        signal_output_dir=FLAGS.signal_output,
                        label_output_dir=FLAGS.label_output,
                        basecall_group=FLAGS.basecall_group,
                        basecall_subgroup=FLAGS.basecall_subgroup,
                        n_thread=FLAGS.threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer fast5 to raw_pair file.')
    parser.add_argument('-i', '--input', required=True,
                        help="Directory that store the fast5 files.")
    parser.add_argument('-so', '--signal_output', required=True)
    parser.add_argument('-lo', '--label_output', required=True)
    parser.add_argument('-t', '--threads', required=False, type=int,
                        help="number of threads")
    parser.add_argument('--basecall_group', default='RawGenomeCorrected_000',
                        help='Basecall group Nanoraw resquiggle \
                        into. Default is Basecall_1D_000')
    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
                        help='Basecall subgroup Nanoraw resquiggle \
                         into. Default is BaseCalled_template')
    parser.add_argument('-raw_len', type=int, default=256, required=True)
    parser.add_argument('-seq_len', type=int, default=70, required=True)
    args = parser.parse_args(sys.argv[1:])
    run(args)
