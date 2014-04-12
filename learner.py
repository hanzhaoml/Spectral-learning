#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.
import numpy as np
from pprint import pprint

class SpectralLearner(object):
    '''This class implements the spectral learning algorithm described in:
            A spectral algorithm for learning hidden markov models
            http://arxiv.org/pdf/0811.4413v6.pdf
    '''
    def __init__(self, num_hidden, num_observ):
        '''
        @num_hidden: np.int, number of the hidden states in HMM.
        @num_observ: np.int, number of the observations in HMM.
        '''
        self._num_hidden = num_hidden
        self._num_observ = num_observ
    
    def train(self, seq):
        '''
        @seq:    np.ndarray, a sequence of observations. Inside the train module, this 
        sequence will be partitioned into several segments.
        
        '''
#        Frequency count of P_1, P_21 and P_3x1
        self._P_1 = np.zeros(self._num_observ, dtype=np.float)
        self._P_21 = np.zeros((self._num_observ, self._num_observ), dtype=np.float)
        self._P_3x1 = [np.zeros((self._num_observ, self._num_observ), dtype=np.float) 
                       for i in xrange(self._num_observ)]
#        Training triples
        trilst = np.array([seq[idx: idx + 3] for idx in xrange(seq.shape[0] - 2)])
        pprint('Number of separated triples: %d' % len(trilst))
        # Parameter estimation
        # Frequency based estimation
        for sq in trilst:
            self._P_1[sq[0]] += 1.0
            self._P_21[sq[1], sq[0]] += 1.0
            self._P_3x1[sq[1]][sq[2], sq[0]] += 1.0
        # Normalization of P_1, P_21, P_3x1
        norms = np.linalg.norm(self._P_1, 1)
        self._P_1 /= norms
        # Normalize the joint distribution of P_21        
        norms = np.sum(self._P_21)
        self._P_21 /= norms
        # Normalize the joint distribution of P_3x1
        norms = 0.0
        for i in xrange(len(self._P_3x1)):
            norms += np.sum(self._P_3x1[i])
        for i in xrange(len(self._P_3x1)):
            self._P_3x1[i] /= norms
        # Singular Value Decomposition
        # Keep all the positive singular values
        (U, _, V) = np.linalg.svd(self._P_21)
        self._U = U[:, 0:self._num_hidden]
        self._V = V[0:self._num_hidden, :]
        # Compute b1, binf and Bx
        # self.factor = (P_21^{T} * U)^{+}, which is used to accelerate the computation
        self._factor = np.linalg.pinv(np.dot(self._P_21.T, self._U))
        self._b1 = np.dot(self._U.T, self._P_1)        
        self._binf = np.dot(self._factor, self._P_1)        
        self._Bx = [np.zeros((self._num_hidden, self._num_hidden), dtype=np.float) 
                    for i in xrange(self._num_observ)]
        for index in xrange(len(self._Bx)):
            self._Bx[index] = np.dot(self._U.T, self._P_3x1[index])
            self._Bx[index] = np.dot(self._Bx[index], self._factor.T)
                   
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
        prob = self._b1
        for ob in seq:
            prob = np.dot(self._Bx[ob], prob)
        prob = np.dot(self._binf.T, prob)
        return prob        
