#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.

from hmm import HMM
from hmm import spectral_learner
from learner import SpectralLearner
import sys
import csv


def print_module(obj):
    print '-' * 50
    print obj
    print '-' * 50
    
def main(modelname, filename):
    hmm = HMM(filename=modelname)
    operator = spectral_learner(hmm)
    sl_learner = SpectralLearner()
    with file(filename, 'r') as f:
        reader = csv.reader(f)
        data = [map(int, row) for row in reader]
    sl_learner.train(data)
    print 'True matrix V:'
    print_module(operator.V)
    print 'Estimated matrix V:'
    print_module(sl_learner.P_21)
    print 'True omega:'
    print_module(operator.w0)
    print 'Estimated omega:'
    print_module(sl_learner.P_1)
    
    
    
    


if __name__ == '__main__':
    usage = '''
    ./experiment.py modelpath train_data
    '''
    if len(sys.argv) < 3:
        print usage
        exit()
    main(sys.argv[1], sys.argv[2])
    
    
    