import os
import argparse
import pandas as pd


def get_report(root_path, assemble):
    report_dic = dict()
    for sample in os.listdir(root_path):
        report_dic[sample] = report_dic.get(sample, {})
        # 14260514
        albacore_path = sample+'-albacore-'+assemble  # 14260514-albacore-canu
        sacall_path = sample+'-sacall-'+assemble
        guppy213_path = sample+'-guppy213-'+assemble
        guppy213ff_path = sample+'-guppy213ff-'+assemble
        guppy213cs_path = sample+'-guppy213cs-'+assemble
        guppy237_path = sample+'-guppy237-'+assemble
        call_list = [sacall_path, albacore_path, guppy213_path,
                     guppy213ff_path, guppy213cs_path, guppy237_path]
        tmp_dic = {}
        for call in call_list:
            call_name = call.split('-')[1]
            tmp_dic[call_name] = tmp_dic.get(call_name, {})
            report_path = root_path+'/'+sample+'/' + \
                call+'/'+'quast_results/latest/report.tsv'
            with open(report_path) as fpath:
                for line in fpath:
                    if any([v in line for v in ['N50', '# misassemblies', 'NA50', '# mismatches per 100 kbp', '# indels per 100 kbp', 'Genome fraction (%)']]):
                        [key, value] = line.strip().split('\t')
                        assert key in ['N50', '# misassemblies', 'NA50',
                                       '# mismatches per 100 kbp', '# indels per 100 kbp', 'Genome fraction (%)'], print(key)
                        tmp_dic[call_name][key] = value
        report_dic[sample] = tmp_dic
    # print(report_dic)
    df_lst = []
    sample_lst = []
    for key, value in report_dic.items():
        df = pd.DataFrame(value, index=['# misassemblies', '# mismatches per 100 kbp',
                                        '# indels per 100 kbp', 'N50',  'NA50', 'Genome fraction (%)'])
        df_lst.append(df)
        sample_lst.append(key)
        # print(df)
    df = pd.concat(df_lst, keys=sample_lst, axis=0)
    df.to_excel(assemble+'-quast-report.xlsx')
    # df = pd.DataFrame(report_dic, index=report_dic.keys())
    # print(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', required=True)
    parser.add_argument('-assembler', required=True, choices=['canu', 'flye'])
    argv = parser.parse_args()
    get_report(argv.root, argv.assembler)


if __name__ == "__main__":
    main()
