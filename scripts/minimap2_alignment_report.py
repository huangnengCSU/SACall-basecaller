import os
import re
import sys


class alignment_record_sam:
    Qname = ''
    flag = 0
    Rname = ''
    qs = 0
    qe = 0
    rs = 0
    re = 0
    map_quality = 0
    mat = 0
    mis = 0
    ins = 0
    dle = 0
    map_len = 0
    cigar = 0

    def convert(self):
        pattern = re.compile(r'((\d)+(S|H|X|=|I|D))')
        it = pattern.finditer(self.cigar)
        i = 0
        read_len = 0
        ref_len = 0
        M, X, I, D, L = 0, 0, 0, 0, 0
        for match in it:
            if match.group().endswith('H') or match.group().endswith('S'):
                length = int(match.group(0)[:len(match.group(0)) - 1])
                if i == 0:
                    self.qs = length
                    i = 1
            elif match.group().endswith('='):
                length = int(match.group(0)[:len(match.group(0)) - 1])
                if i == 0:
                    self.qs = 0
                    i = 1
                M += length
                L += length
                #				if length >= 15:
                #					print read_len, '\t', ref_len, '\t', length
                read_len += length
                ref_len += length
            elif match.group().endswith('X'):
                length = int(match.group(0)[:len(match.group(0)) - 1])
                if i == 0:
                    self.qs = 0
                    i = 1
                X += length
                L += length
                read_len += length
                ref_len += length
            elif match.group().endswith('D'):
                length = int(match.group(0)[:len(match.group(0)) - 1])
                if i == 0:
                    self.qs = 0
                    i = 1
                ref_len += length
                D += length
                L += length
            elif match.group().endswith('I'):
                length = int(match.group(0)[:len(match.group(0)) - 1])
                if i == 0:
                    self.qs = 0
                    i = 1
                read_len += length
                I += length
                L += length
        self.qe = self.qs + read_len
        self.re = self.rs + ref_len
        self.mat = M
        self.mis = X
        self.ins = I
        self.dle = D
        self.map_len = L


def get_alignment_record(sam):
    fsam = open(sam, 'r')
    record_list = []
    for line in fsam:
        if line.startswith('@'):
            continue
        else:
            records = alignment_record_sam()
            line = line.strip('\n')
            line_items = line.split('\t')
            records.Qname = line_items[0]
            records.flag = int(line_items[1])
            records.Rname = line_items[2]
            records.rs = int(line_items[3]) - 1
            records.map_quality = line_items[4]
            records.cigar = line_items[5]

            records.convert()
            record_list.append(records)
    fsam.close()
    return record_list


def main():
    allrecords = get_alignment_record(sys.argv[1])
    print(
        'qname\trname\tmap quality\tflag\tqs\tqe\trs\tre\tmatch\tmismatch\tinsertion\tdeletion\tidentity rate\terror rate\tmap length')
    for record in allrecords:
        if record.flag != 4:
            print(record.Qname, '\t', record.Rname, '\t', record.map_quality, '\t', record.flag, '\t', record.qs, '\t',
                  record.qe, '\t', record.rs, '\t', record.re, '\t', record.mat * 1.0 / (record.re - record.rs), '\t',
                  record.mis * 1.0 / (record.re - record.rs), '\t', record.ins * 1.0 / (record.re - record.rs),
                  '\t', record.dle * 1.0 / (record.re - record.rs), '\t',
                  1 - record.dle * 1.0 / (record.re - record.rs) - record.mis * 1.0 / (record.re - record.rs), '\t',
                  record.mis * 1.0 / (record.re - record.rs) + record.ins * 1.0 / (
                              record.re - record.rs) + record.dle * 1.0 / (record.re - record.rs), '\t', record.map_len)


if __name__ == '__main__':
    main()