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
from hmm import spectral_learner
from hmm import HMM

class SpectralLearner(object):
    '''This class implements the spectral learning algorithm described in:
            A spectral algorithm for learning hidden markov models
            http://arxiv.org/pdf/0811.4413v6.pdf
    '''
    def __init__(self):
        pass
    
    def train(self, trilst, m):
        '''
        @trilst:    list[], list of observation sequences
        Frequency based estimation of P_1, P_21 and P_3x1, where x = 1, 2, ... n, n is the
        size of observation states.
        '''
        self.m = m
        print 'Estimating n from observable data...'
        self.n = max(set([ob for sublist in trilst for ob in sublist]))
#        Frequency count of P_1, P_21 and P_3x1
        self.P_1 = np.zeros(self.n, dtype=np.float)
        self.P_21 = np.zeros((self.n, self.n), dtype=np.float)
        self.P_3x1 = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
#        Training triples
        print 'Number of input sequences:', len(trilst)
        trilst = [sublst[idx: idx + 3] for sublst in trilst for idx in xrange(len(sublst) - 2)]
        print 'Number of separated triples:', len(trilst)
        print 'Estimate P_1, P_21 and P_3x1...'
        # Parameter estimation
        # Frequency based estimation
        for sq in trilst:
            self.P_1[sq[0]] += 1.0
            self.P_21[sq[1], sq[0]] += 1.0
            self.P_3x1[sq[1]][sq[2], sq[0]] += 1.0
#        Laplacian smoothing, doesn't change anything
#        self.P_1 += 1.0
#        self.P_21 += 1.0
#        for operator in self.P_3x1:
#            operator += 1.0
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
        

class BoostedLearner(object):
    '''
    Using AdaBoost.RT to boost the basic SpectralLearner, the goal is
    to relieve the negative probability problem as well as solve the 
    model-selection problem
    '''
    def __init__(self):
        pass
    
    def _encode_triple(self, triple):
        '''
        @triple:    list of three int number
        In order to facilitate the building of element-wise multinomial distribution,
        we need a way to encode each unique triple into a unique integer
        '''
        hash_value = 0
        for elem in triple:
            hash_value *= self.codec
            hash_value += elem
        return hash_value
        
    def _decode_triple(self, hash_value):
        '''
        @hash_value:    int
        Corresponding to the encode_triple procedure, this function is used to decode each
        unique integer to its corresponding triple
        '''
        triple = []
        while hash_value != 0:
            triple.append(hash_value % self.codec)
            hash_value /= self.codec
        triple.reverse()
        return triple
        
    def _build_distribution(self, trilist):
        '''
        @trilist:    list[triple],    list of triples
        Build distribution based on the given list of triples
        '''
        codec_triples = [self._encode_triple(triple) for triple in trilist]
        uniform_weight = 1.0 / len(codec_triples)
        distribution = {key: 0.0 for key in set(codec_triples)}
#        Build distribution on the assumption that prior distribution for each triple
#        obeys the uniform distribution
        for key in codec_triples:
            distribution[key] += uniform_weight
        decoded_triples = [self._decode_triple(hash_value) for hash_value in distribution.keys()]
        weights = distribution.values()
        return (decoded_triples, weights)
                
    def _train(self, triples, weights, size, m):
        '''
        @weights:    np.array, weights of each triple in trilist
        @trilist:    list[list], list of triples used to train each spectral learner
        
        Return a weak regressor based on the current configuration of weighting and 
        the number of hidden states m 
        '''
        counts = np.random.multinomial(size, weights, size=1).tolist()
#        Frequency count of P_1, P_21 and P_3x1
        P_1 = np.zeros(self.n, dtype=np.float)
        P_21 = np.zeros((self.n, self.n), dtype=np.float)
        P_3x1 = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
#        Training triples
        # Parameter estimation
        # Frequency based estimation
        for idx, count in enumerate(counts):
            triple = triples[idx]
            P_1[triple[0]] += count
            P_21[triple[1], triple[0]] += count
            P_3x1[triple[1]][triple[2], triple[0]] += count
        # Normalization of P_1, P_21, P_3x1
        norms = np.linalg.norm(P_1, 1)
        P_1 /= norms
        # Normalize the joint distribution of P_21        
        norms = np.sum(P_21)
        P_21 /= norms
        # Normalize the joint distribution of P_3x1
        norms = 0.0
        for i in xrange(len(P_3x1)):
            norms += np.sum(P_3x1[i])
        for i in xrange(len(P_3x1)):
            P_3x1[i] /= norms
        # Singular Value Decomposition
        # Keep all the positive singular values
        (U, S, V) = np.linalg.svd(P_21)
        U = U[:, 0: m]
        V = V[0: m, :]
        # Compute b1, binf and Bx
        # self.factor = (P_21^{T} * U)^{+}, which is used to accelerate the computation
        factor = np.linalg.pinv(np.dot(P_21.T, U))
        b1 = np.dot(U.T, P_1)        
        binf = np.dot(factor, P_1)        
        Bx = [np.zeros((m, m), dtype=np.float) for i in xrange(self.n)]
        for index in xrange(len(Bx)):
            Bx[index] = np.dot(U.T, P_3x1[index])
            Bx[index] = np.dot(Bx[index], factor.T)
        return _ObservableOperator(b1, binf, Bx)
            
    def boosting(self, observations):
        '''
        @observations:    list[list], list of observation sequences used to train 
                        boosted learner
        '''
#        Estimating the number of observations from data
        self.n = max(set([ob for seq in observations for ob in seq]))
        self.codec = self.n + 1
#        Split each long observation sequence into several observation triples
        trilist = [seq[i: i+3] for seq in observations for i in xrange(len(seq)-2)]
        (triples, weights) = self._build_distribution(trilist) 
#        Rounds of boosting
        _T = self.n
        classifiers, voting_weights = [], []
        round = 0
        while round < _T:
#            For each round, set the model parameter m = round+1
            classifiers[round] = self._train(triples, weights, len(trilist), round+1)
#            Sadly, since the problem of learning a HMM is under the range of 
#            unsupervised learning, we cannot use the technique of AdaBoosting to 
#            improve the performance
            round += 1
            

class _ObservableOperator(object):
    '''
    Observable Operator used in spectral learning algorithm:
    b_1, b_inf and Bx for each x in Observation sets
    '''
    def __init__(self, b1, binf, Bx):
        self.b1 = b1
        self.binf = binf
        self.Bx = Bx
        
    def predict(self, seq):
        '''
        @seq:    list, observation sequence
        return the predicted joint probability of this observation probability
        '''
        # binf * B_{xt:1} * b1
        prob = self.b1
        for ob in seq:
            prob = np.dot(self.Bx[ob], prob)
        prob = np.dot(self.binf.T, prob)
        return prob
        
    

def main(modelpath, trainfile, testfile, m):
    with file(trainfile, 'r') as fin:
        reader = csv.reader(fin)
        training_data = [map(int, row) for row in reader]
    with file(testfile, 'r') as fin:
        reader = csv.reader(fin)
        testing_data = [map(int, row) for row in reader]
    sp_learner = SpectralLearner()
    sp_learner.train(training_data, m)    
    hmm = HMM(filename=modelpath)
    print 'Observation probability for sequence in training data set'
    print '*' * 50
    for seq in training_data:
        print hmm.probability(seq), '\t', sp_learner.predict(seq), '\t', seq
    print '=' * 50
    print 'Observation probability for sequence in testing data set'
    print '*' * 50
    for seq in testing_data:
        print hmm.probability(seq), '\t', sp_learner.predict(seq), '\t', seq


if __name__ == '__main__':
    usage = '''
        usage: ./learner.py modelpath trainfile testfile m(Model selection)
    '''
    if len(sys.argv) < 5:
        print usage
        exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    
        
        
        
        
