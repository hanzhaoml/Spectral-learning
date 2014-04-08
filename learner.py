#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.

import numpy as np
import sys
import csv
from pprint import pprint
from hmm import spectral_learner
from hmm import HMM

class SpectralLearner(object):
    '''This class implements the spectral learning algorithm described in:
            A spectral algorithm for learning hidden markov models
            http://arxiv.org/pdf/0811.4413v6.pdf
    '''
    def __init__(self):
        pass
    
    def train(self, seq, m, n):
        '''
        @seq:    list[], list of observation sequences
        Frequency based estimation of P_1, P_21 and P_3x1, where x = 1, 2, ... n, n is the
        size of observation states.
        '''
        self.m = m
        self.n = n
#        Frequency count of P_1, P_21 and P_3x1
        self.P_1 = np.zeros(self.n, dtype=np.float)
        self.P_21 = np.zeros((self.n, self.n), dtype=np.float)
        self.P_3x1 = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
#        Training triples
        trilst = np.array([seq[idx: idx+3] for idx in xrange(seq.shape[0]-2)])
        print 'Number of separated triples:', len(trilst)
        print 'Estimate P_1, P_21 and P_3x1...'
        # Parameter estimation
        # Frequency based estimation
        for sq in trilst:
            self.P_1[sq[0]] += 1.0
            self.P_21[sq[1], sq[0]] += 1.0
            self.P_3x1[sq[1]][sq[2], sq[0]] += 1.0
        print 'Normalizing P_1, P_21 and P_3x1...'
        # Normalization of P_1, P_21, P_3x1
        norms = np.linalg.norm(self.P_1, 1)
        self.P_1 /= norms
        # Normalize the joint distribution of P_21        
        norms = np.sum(self.P_21)
        self.P_21 /= norms
        # Normalize the joint distribution of P_3x1
        norms = 0.0
        for i in xrange(len(self.P_3x1)):
            norms += np.sum(self.P_3x1[i])
        for i in xrange(len(self.P_3x1)):
            self.P_3x1[i] /= norms
        # Singular Value Decomposition
        # Keep all the positive singular values
        (U, S, V) = np.linalg.svd(self.P_21)
        self.U = U[:, 0:self.m]
        self.V = V[0:self.m, :]
        # Compute b1, binf and Bx
        # self.factor = (P_21^{T} * U)^{+}, which is used to accelerate the computation
        self.factor = np.linalg.pinv(np.dot(self.P_21.T, self.U))
        self.b1 = np.dot(self.U.T, self.P_1)        
        self.binf = np.dot(self.factor, self.P_1)        
        self.Bx = [np.zeros((self.m, self.m), dtype=np.float) for i in xrange(self.n)]
        for index in xrange(len(self.Bx)):
            self.Bx[index] = np.dot(self.U.T, self.P_3x1[index])
            self.Bx[index] = np.dot(self.Bx[index], self.factor.T)
        print 'End training...'
                   
    def predict(self, seq):
        '''
        @lst:
            type:    numpy.array
            param:   Observation sequence.
        
        @return:
            type:    float
            param:   The probability of the observation sequence estimated by 
                     the learned model
                    
                     P[x_t+1, x_t, x_t-1, ... x1]
        '''
        # binf * B_{xt:1} * b1
        prob = self.b1
        for ob in seq:
            prob = np.dot(self.Bx[ob], prob)
        prob = np.dot(self.binf.T, prob)
        return prob    


def main(modelpath, trainfile, testfile, m, n):
    with file(trainfile, 'r') as fin:
        reader = csv.reader(fin)
        training_data = [map(int, row) for row in reader]
    with file(testfile, 'r') as fin:
        reader = csv.reader(fin)
        testing_data = [map(int, row) for row in reader]
    sp_learner = SpectralLearner()
    sp_learner.train(training_data, m, n)    
    hmm = HMM(filename=modelpath)
    pprint('Observation probability for sequence in training data set')
    print '*' * 50
    for seq in training_data:
        print hmm.probability(seq), '\t', sp_learner.predict(seq), '\t', seq
    pprint('=' * 50)
    pprint('Observation probability for sequence in testing data set')
    pprint('*' * 50)
    for seq in testing_data:
        print hmm.probability(seq), '\t', sp_learner.predict(seq), '\t', seq


if __name__ == '__main__':
    usage = '''
        usage: ./learner.py modelpath trainfile testfile m(Model selection) 
        n(Number of possible observations)
    '''
    if len(sys.argv) < 5:
        print usage
        exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    