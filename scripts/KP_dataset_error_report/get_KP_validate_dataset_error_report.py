import os
import argparse
import pandas as pd


def get_report(root_path):
    report_dic = dict()
    for sample in os.listdir(root_path):
        report_dic[sample] = report_dic.get(sample, {})
        # 14260514
        sacall_error = root_path+'/'+sample+'/'+'sacall_error.txt'
        albacore_error = root_path+'/'+sample+'/'+'albacore_error.txt'
        guppy213_error = root_path+'/'+sample+'/'+'guppy213_error.txt'
        guppy213ff_error = root_path+'/'+sample+'/'+'guppy213ff_error.txt'
        guppy213cs_error = root_path+'/'+sample+'/'+'guppy213cs_error.txt'
        guppy237_error = root_path+'/'+sample+'/'+'guppy237_error.txt'
        error_list = [sacall_error,albacore_error,guppy213_error,guppy213ff_error,guppy213cs_error,guppy237_error]
        tmp_dic={}
        for error_file in error_list:
            call_name = os.path.basename(error_file).split('_')[0]
            tmp_dic[call_name]=tmp_dic.get(call_name,{})
            with open(error_file,'r') as f:
                error_info = f.readline().strip('\t').split('\t')
                tmp_dic[call_name]['deletion rate']=error_info[1]
                tmp_dic[call_name]['insertion rate']=error_info[2]
                tmp_dic[call_name]['mismatch rate']=error_info[3]
                tmp_dic[call_name]['total error']=error_info[4]
        report_dic[sample] = tmp_dic
    df_lst = []
    sample_lst = []
    for key, value in report_dic.items():
        df = pd.DataFrame(value, index=['deletion rate', 'insertion rate','mismatch rate', 'total error'],
                                columns=['sacall','guppy213cs','guppy213','guppy213ff','guppy237','albacore']).T
        df_lst.append(df)
        sample_lst.append(key)
        # print(df)
    # print(sample_lst)
    df = pd.concat(df_lst, keys=sample_lst, axis=0)
    df.to_excel('alignment-error-report.xlsx')
    # # df = pd.DataFrame(report_dic, index=report_dic.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', required=True)
    argv = parser.parse_args()
    get_report(argv.root)


if __name__ == "__main__":
    main()
