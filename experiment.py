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
import math

import numpy as np

from hmm import HMM
from learner import SpectralLearner

class Experimenter(object):
    '''
    This class is built to facilitate the experiments of different learning
    algorithms.
    '''
    def __init__(self, trainfile, testfile, modelpath):
        self.load_training(trainfile)
        self.load_testing(testfile)
        self.load_model(modelpath)
    
    def load_training(self, filename):
        '''
        @filename:    string, path of the training file
        Load training data into experimenter.
        The training data should contain lists of observation sequence,
        each sequence is composed of several observations and different 
        observations are separated by commas.
        '''
        with file(filename, 'r') as fin:
            reader = csv.reader(fin)
            self.training_data = [map(int, seq) for seq in reader]
        
    def load_testing(self, filename):
        '''
        @filename:    string, path of the testing file
        Load testing data into experimentor.
        The testing data should contain lists of observation sequence,
        each sequence is composed of several observations and different
        observations are separated by commas.
        '''
        with file(filename, 'r') as fin:
            reader = csv.reader(fin)
            self.testing_data = [map(int, seq) for seq in reader]
    
    def load_model(self, modelpath):
        '''
        @modelpath:    string, path of the model
        Load the parameters of a hidden markov model
        '''
        self._model = HMM.from_file(modelpath)

    def _train(self):
        '''
        Train a Hidden Markov Model with differnt learning algorithms
        '''
        t_start = time.time()
        self._sl_learner = SpectralLearner()
        self._sl_learner.train(self.training_data, 1, self._model.n)
        t_end = time.time()
        print 'Time used for Spectral learner and OOM learner:', (t_end - t_start)

    def testing(self, outfile):
        '''
        @outfile:    string, filepath of the output log
        '''
        variation_dist = 0.0
        self._train()
        records = list()
        for seq in self.testing_data:
            model_prob = self._model.probability(seq)
            sl_prob = self._sl_learner.predict(seq)
            records.append((model_prob, sl_prob))
            variation_dist += abs(model_prob-sl_prob)
            
        with file(outfile, 'w') as out:
            out.write('The value of m used in spectral learning algorithm: %d\n' % self._sl_learner.m)
            out.write('Model probability\tSpectral learning probability\n')
            for idx, record in enumerate(records):
                line = '%e\t%e\t%s\n' %(record[0], record[1], self.testing_data[idx])
                out.write(line)
        
        out.write("-" * 50)
        out.write("Variation dist: %f" % variation_dist)
    
    
def main(trainfile, testfile, modelpath):
    experimenter = Experimenter(trainfile, testfile, modelpath)
    experimenter.testing('experiment.log')
    
if __name__ == '__main__':
    usage = '''
    ./experiment.py train_data test_data model_path
    '''
    if len(sys.argv) < 4:
        print usage
        exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    