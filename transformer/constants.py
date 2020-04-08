#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename: 
# @Date: 2018-12-05 16:22
# @author: huangneng
# @contact: huangneng@csu.edu.cn

SIG_PAD = -10
PAD = 0
BOS = 6
EOS = 7

A = 1
T = 2
C = 3
G = 4
U = 5

PAD_WORD = '<blank>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

BASE_DIC = {'A': A, 'T': T, 'C': C, 'G': G, 'U': U,
            PAD_WORD: PAD, BOS_WORD: BOS, EOS_WORD: EOS}

TRANSLATE_DIC = {A: 'A', T: 'T', C: 'C', G: 'G',
                 U: 'U', PAD: PAD_WORD, BOS: BOS_WORD, EOS: EOS_WORD}