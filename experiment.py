#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.
import sys
import time
import csv
import numpy as np

from pprint import pprint
from hmm import HMM
from hmm import EMHMM
from hmm import SLHMM


class Experimenter(object):
    '''
    This class is built to facilitate the experiments of different learning
    algorithms.
    '''
    def __init__(self, training_filename, test_filename, model_filename, num_hidden,
                 num_observ, num_em_restarts=20):
        self._training_data = [np.loadtxt(training_filename, dtype=np.int, delimiter=",")]
        # self._test_data = np.loadtxt(test_filename, dtype=np.int, delimiter=",")
        self._test_data = []
        with file(test_filename, "rb") as fin:
            reader = csv.reader(fin)
            for line in reader:
                self._test_data.append(np.asarray(map(int, line)))
        self._model = HMM.from_file(model_filename)
        self._num_hidden = num_hidden
        self._num_observ = num_observ
        self._num_em_restarts = num_em_restarts
    
    @property
    def training_size(self):
        return self._training_data[0].shape[0]
    
    @property
    def test_size(self):
        return len(self._test_data)

    @property
    def num_em_restarts(self):
        return self._num_em_restarts
                
    def _train(self, num_train_inst):
        '''
        Train a Hidden Markov Model with differnt learning algorithms
        '''
        num_train_inst = min(num_train_inst, self._training_data[0].shape[0])
        training_data = self._training_data[0][:num_train_inst]
        pprint("=" * 50)
        pprint("Training set length: %d" % num_train_inst)
        # Spectral learning algorithm
        start_time = time.time()
        self._sl_learner = SLHMM(self._num_hidden, self._num_observ)
        self._sl_learner.fit([training_data])
        end_time = time.time()
        pprint("Time used for Spectral Learner: %f" % (end_time - start_time))
        sl_time = end_time - start_time
        # Expectation Maximization algorithm
        #self._em_learners = []
        em_times = np.zeros(self._num_em_restarts, dtype=np.float)
        #for i in xrange(self._num_em_restarts):
            #self._em_learners.append(EMHMM(self._num_hidden, self._num_observ))
            #start_time = time.time()
            #self._em_learners[i].fit([training_data], max_iters=20, verbose=True)
            #end_time = time.time()
            #pprint("Time used for Expectation Maximization: %f" % (end_time - start_time))
            #em_times[i] = end_time - start_time
        return (sl_time, np.mean(em_times))
    
    def run_experiment(self, num_train_inst):
        '''
        @log_filename:    string, filepath of the output log
        '''
        sl_time, em_time = self._train(num_train_inst)
        true_probs = np.zeros(len(self._test_data), dtype=np.float)
        sl_probs = np.zeros(len(self._test_data), dtype=np.float)
        em_probs = np.zeros((self._num_em_restarts, len(self._test_data)), dtype=np.float)
        for i, seq in enumerate(self._test_data):
            true_probs[i] = self._model.predict(seq)
            sl_probs[i] = self._sl_learner.predict(seq)
            #for j in xrange(self._num_em_restarts):
                #em_probs[j, i] = self._em_learners[j].predict(seq)
        # L1-distance between true probability and inference probability by spectral learning
        sl_variation_dist = np.abs(true_probs - sl_probs)
        # L1-distance between true probability and inference probability by expectation maximization
        em_variation_dist = np.abs(true_probs - em_probs)
        # Sum of L1-distance
        sl_variation_measure = np.sum(sl_variation_dist)
        em_variation_measure = np.sum(em_variation_dist, axis=1)
        return (sl_time, em_time, sl_variation_measure, em_variation_measure)
    
    
def compare_with_em(trainfile, testfile, modelpath, num_hidden, num_observ, log_filename):
    experimenter = Experimenter(trainfile, testfile, modelpath, 
                                num_hidden, num_observ)
    chunk_size = 1000
    num_train_chunks = experimenter.training_size / chunk_size
    num_train_insts = chunk_size * np.arange(1, num_train_chunks+1)
    # Number of Columns needed = num_em_restart + sl_prob + em_train_time + sl_train_time + num_training_inst
    #                          = num_em_restart + 4
    statistics = np.zeros((num_train_chunks, experimenter.num_em_restarts+4), dtype=np.float)
    for i, num_train_inst in enumerate(num_train_insts):
        statistics[i, 0] = num_train_inst
        (sl_time, em_time, sl_variation_measure, em_variation_measure) = experimenter.run_experiment(num_train_inst)
        statistics[i, 1] = sl_time
        statistics[i, 2] = em_time
        statistics[i, 3] = sl_variation_measure
        statistics[i, 4:] = em_variation_measure
    np.savetxt(log_filename, statistics, delimiter=",", fmt="%e")
    
def model_selection(trainfile, testfile, modelpath, log_filename):
    training_data = np.loadtxt(trainfile, dtype=np.int, delimiter=",")
    test_data = []
    with file(testfile, "rb") as fin:
        reader = csv.reader(fin)
        for line in reader:
            test_data.append(np.asarray(map(int, line)))
    model = HMM.from_file(modelpath)
    num_hidden = model.m
    num_observ = model.n
    variation_measure = np.zeros(num_observ, dtype=np.float)
    neg_num_measure = np.zeros(num_observ, dtype=np.int)
    neg_proportion_measure = np.zeros(num_observ, dtype=np.float)
    for m in xrange(1, num_observ + 1):
        slearner = SpectralLearner()
        slearner.train(training_data, m, num_observ)
        true_probs = np.zeros(len(test_data))
        sl_probs = np.zeros(len(test_data))
        for i, seq in enumerate(test_data):
            true_probs[i] = model.probability(seq)
            sl_probs[i] = slearner.predict(seq)
            # pprint("%e %e" % (true_probs[i], sl_probs[i]))
        neg_num_measure[m - 1] = np.sum(sl_probs < 0, dtype=np.float)
        neg_proportion_measure[m - 1] = neg_num_measure[m - 1] / float(len(test_data))
        partition_function = np.sum(true_probs)
        #Normalizing joint probability distribution to get conditional distribution
        true_probs /= partition_function
        sl_probs /= partition_function
        variation_measure[m - 1] = np.sum(np.abs(sl_probs - true_probs))
        pprint("Model Rank Hyperparameter: %d" % m)
        pprint("Sum of all true probabilities: %f" % np.sum(true_probs))
        pprint("Sum of all estimated probabilities: %f" % np.sum(sl_probs))
        pprint("*" * 50)
    statistics = np.array([variation_measure, neg_num_measure, neg_proportion_measure])
    statistics = statistics.T
    np.savetxt(log_filename, statistics, delimiter=",", fmt="%e")
    
    
if __name__ == '__main__':
    usage = '''
    ./experiment.py training_filename test_filename model_filename log_filename
    num_hidden num_observ
    '''
    if len(sys.argv) < 7:
        print usage
        exit()
    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    model_filename = sys.argv[3]
    log_filename = sys.argv[4]
    #model_selection(training_filename, test_filename, model_filename, log_filename)
    num_hidden = int(sys.argv[5])
    num_observ = int(sys.argv[6])
    compare_with_em(training_filename, test_filename, model_filename, 
                    num_hidden, num_observ, log_filename)
