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

from hmm import HMM

def generate(m, n, dsize, tsize, filename):
    t_start = time.clock()
    
    # Create Hidden Markov Model randomly. 
    # m --- The number of hidden states
    # n --- The number of observation states
    model = HMM(m, n)

    # Generate training set and testing set
    training_seq = model.generate_train_data(dsize)
#     test_seqs = model.generate_test_data(int(sys.argv[5]))
    test_seqs = model.gendata(tsize, 50)
    
    training_filename = "train_" + filename + ".data"
    test_filename = "test_" + filename + ".data"
    model_filename = "model_" + filename + ".npy"
    
    np.savetxt(training_filename, [training_seq], delimiter=",", fmt="%d")
#     np.savetxt(test_filename, test_seqs, delimiter=",", fmt="%d")
    with file(test_filename, "wb") as fout:
        writer = csv.writer(fout)
        for seq in test_seqs:
            writer.writerow(seq)
    
    t_end = time.clock()
    print 'Time used:', (t_end - t_start), 'seconds'
    
    # Save model parameters
    HMM.to_file(model_filename, model)    
    
    
def regenerate(hmm, training_filename, dsize):
    training_seq = hmm.generate_train_data(dsize)
    np.savetxt(training_filename, [training_seq], delimiter=",", fmt="%d")
    
if __name__ == '__main__':    
    usage = '''./generator.py file m n training_seq_length testing_seq_length
            file is the filepath to store the data 
            m is the size of states in HMM    
            n is the size of observations in HMM  
            dsize is the size of data you\'d like to generate    
            sqlen is the maximal length of each observation sequence.\n'''
#     if len(sys.argv) < 6:
#         sys.stderr.write(usage)
#         exit()
    if len(sys.argv) < 4:
        sys.stderr.write("Need more parameters")
        exit()
    hmm = HMM.from_file(sys.argv[1])
    regenerate(hmm, sys.argv[2], int(sys.argv[3]))
