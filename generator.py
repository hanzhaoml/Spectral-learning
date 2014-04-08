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

if __name__ == '__main__':    
    usage = '''./generator.py file m n training_seq_length testing_seq_length
            file is the filepath to store the data 
            m is the size of states in HMM    
            n is the size of observations in HMM  
            dsize is the size of data you\'d like to generate    
            sqlen is the maximal length of each observation sequence.\n'''
    if len(sys.argv) < 6:
        sys.stderr.write(usage)
        exit()
    
    t_start = time.clock()
    
    # Create Hidden Markov Model randomly. 
    # m --- The number of hidden states
    # n --- The number of observation states
    model = HMM(int(sys.argv[2]), int(sys.argv[3]))

    # Generate training set and testing set
    training_seq = model.generate_train_data(int(sys.argv[4]))
#     test_seqs = model.generate_test_data(int(sys.argv[5]))
    test_seqs = model.gendata(int(sys.argv[5]), 20)
    
    training_filename = "train_" + sys.argv[1] + ".data"
    test_filename = "test_" + sys.argv[1] + ".data"
    model_filename = "model_" + sys.argv[1] + ".npy"
    
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