#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.
import sys
import time
import numpy as np

from pprint import pprint
from hmm import HMM
from EM import BaumWelch
from learner import SpectralLearner

class Experimenter(object):
    '''
    This class is built to facilitate the experiments of different learning
    algorithms.
    '''
    def __init__(self, training_filename, test_filename, model_filename, model_parameter):
        self._training_data = np.loadtxt(training_filename, dtype=np.int, delimiter=",")
        self._test_data = np.loadtxt(test_filename, dtype=np.int, delimiter=",")
        self._model = HMM.from_file(model_filename)
        self._parameter = model_parameter
        
    def _train(self, num_train_inst):
        '''
        Train a Hidden Markov Model with differnt learning algorithms
        '''
        num_train_inst = min(num_train_inst, self._training_data.shape[0])
        training_data = self._training_data[:num_train_inst]
        # Spectral learning algorithm
        start_time = time.time()
        self._sl_learner = SpectralLearner()
        self._sl_learner.train(training_data, self._parameter, self._model.n)
        end_time = time.time()
        pprint("Time used for Spectral Learner: %f" % (end_time - start_time))
        # Expectation Maximization algorithm
        start_time = time.time()
        self._em_learner = BaumWelch(self._parameter, self._model.n)
        self._em_learner.train(training_data)
        end_time = time.time()
        pprint("Time used for Expectation Maximization: %f" % (end_time - start_time))
        
    def run_experiment(self, num_train_inst, log_filename):
        '''
        @log_filename:    string, filepath of the output log
        '''
        pprint("Length of training data: %d" % self._training_data.shape[0])
        self._train(num_train_inst)
        true_probs = np.zeros(self._test_data.shape[0], dtype=np.float)
        sl_probs = np.zeros(self._test_data.shape[0], dtype=np.float)
        em_probs = np.zeros(self._test_data.shape[0], dtype=np.float)
        for i, seq in enumerate(self._test_data):
            true_probs[i] = self._model.probability(seq)
            sl_probs[i] = self._sl_learner.predict(seq)
            em_probs[i] = self._em_learner.predict(seq)
        # L1-distance between true probability and inference probability by spectral learning
        sl_variation_dist = np.abs(true_probs - sl_probs)
        # L1-distance between true probability and inference probability by expectation maximization
        em_variation_dist = np.abs(true_probs - em_probs)
        # Percentage of negative probabilities:
        neg_percentage = np.sum(sl_probs < 0, dtype=np.float) / sl_probs.shape[0]
        num_neg_probs = np.sum(sl_probs < 0)
        with file(log_filename, "wb") as fout:
            fout.write('The value of m used in spectral learning algorithm: %d\n' % self._parameter)
            fout.write('Truth\tSL\t EM\t OEM\n')
            for i in xrange(self._test_data.shape[0]):
                line = '%e\t%e\t%e\t%s\n' % (true_probs[i], sl_probs[i], em_probs[i], self._test_data[i, :])
                fout.write(line)        
            fout.write("-" * 50 + "\n")
            line = "%f\t%f\t%f\n" % (np.sum(true_probs), np.sum(sl_probs), np.sum(em_probs))
            fout.write(line)
        pprint("Percentage of negative probabilities: %f" % neg_percentage)
        pprint("Number of negative probabilies: %d" % num_neg_probs)
        pprint("Variation distance for Spectral Learning: %f" % np.sum(sl_variation_dist))
        pprint("Variation distance for Expectation Maximization: %f" % np.sum(em_variation_dist))        
        
    
def main(trainfile, testfile, modelpath, model_parameter):
    experimenter = Experimenter(trainfile, testfile, modelpath, model_parameter)
    num_train_insts = [2000, 4000, 6000, 8000, 10000]
#     num_train_insts = [2000]
    for num_train_inst in num_train_insts:
        experimenter.run_experiment(num_train_inst, 'experiment.log')
        pprint("-" * 50)
        
        
if __name__ == '__main__':
    usage = '''
    ./experiment.py train_data test_data model_path model_parameter
    '''
    if len(sys.argv) < 5:
        print usage
        exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    
