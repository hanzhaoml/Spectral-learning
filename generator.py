#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.
import sys
import csv
import time
import numpy as np
from pprint import pprint
from hmm import HMM

def generate(m, n, dsizes, tsize, file_tager):
    start_time = time.time()
    # Create Hidden Markov Model randomly. 
    # m --- The number of hidden states
    # n --- The number of observation states
    model = HMM(m, n)
    # Generate training set
    for dsize in dsizes:
        training_seq = model.generate_train_data(dsize)
        training_filename = "train_" + file_tager + ("_%d.data" % dsize)
        np.savetxt(training_filename, [training_seq], delimiter=",", fmt="%d")
    # Generate test set
    test_seqs_F = model.generate_test_data(tsize, min_seq_len=4, max_seq_len=5)
    test_seqs_V = model.generate_test_data(tsize, min_seq_len=3, max_seq_len=50)

    test_seqs_F_filename = "test_" + file_tager + "_F.data"
    test_seqs_V_filename = "test_" + file_tager + "_V.data"

    with file(test_seqs_F_filename, "wb") as fout:
        writer = csv.writer(fout)
        for seq in test_seqs_F:
            writer.writerow(seq)

    with file(test_seqs_V_filename, "wb") as fout:
        writer = csv.writer(fout)
        for seq in test_seqs_V:
            writer.writerow(seq)
    end_time = time.time()
    pprint("Time used: %f seconds" % (end_time-start_time))
    # Save model to file 
    model_filename = "model_" + file_tager + ".npy"
    HMM.to_file(model_filename, model)    


def regenerate_training(hmm, training_filename, dsize):
    training_seq = hmm.generate_train_data(dsize)
    np.savetxt(training_filename, [training_seq], delimiter=",", fmt="%d")

def regenerate_test(hmm, test_filename, dsize, max_length=50):
    test_seqs = hmm.generate_test_data(tsize, min_seq_len=50, max_seq_len=51)
    with file(test_filename, "wb") as fout:
        writer = csv.writer(fout)
        for seq in test_seqs:
            writer.writerow(seq)

if __name__ == '__main__':    
    usage = '''./generator.py file m n test_set_size 
            file is the filepath to store the data 
            m is the size of states in HMM    
            n is the size of observations in HMM  
            test_set_size is the size of test_set
            '''
    if len(sys.argv) < 5:
        sys.stderr.write(usage)
        exit()
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    dsizes = [10000, 50000, 100000, 500000, 1000000, 5000000]
    tsize = int(sys.argv[4])
    file_tager = sys.argv[1]
    #generate(m, n, dsizes, tsize, file_tager)
    hmm = HMM.from_file(sys.argv[1])
    #regenerate_training(hmm, sys.argv[2], int(sys.argv[3]))
    regenerate_test(hmm, "m4n8_50_len_test.data", tsize)
