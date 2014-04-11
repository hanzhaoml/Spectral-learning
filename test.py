#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.

import time
import sys
import numpy as np

from pprint import pprint
from hmm import HMM
from EM import BaumWelch

def testBaumWelch(training_filename, test_filename, model_filename):
    '''
    @training_filename: string. Path to training set
    @test_filename: string. Path to test set
    @model_filename: string. Path to the HMM which generates the training
                     and test set
    '''
    hmm = HMM.from_file(model_filename)
    learner = BaumWelch(hmm.m, hmm.n)
    training_data = np.loadtxt(training_filename, dtype=np.int, delimiter=",")
    test_data = np.loadtxt(test_filename, dtype=np.int, delimiter=",")
    pprint("Start training HMM with EM...")
    start_time = time.clock()
    learner.train(training_data)
    end_time = time.clock()
    pprint("Finished training HMM with EM...")
    pprint("Total time used to train HMM with EM: %f" % (end_time-start_time))
    
if __name__ == '__main__':
    usage = './test training_filename, test_filename, model_filename'
    if len(sys.argv) < 4:
        print usage
        exit()
    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    model_filename = sys.argv[3]
    testBaumWelch(training_filename, test_filename, model_filename) 
    