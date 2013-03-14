#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.


#Based on OOM theory to learn a Hidden Markov Model,
#details about OOM theory can be found at:
#Observable operator models for discrete stochastic time series
import numpy as np


class OOMLearner(object):
    '''
    This class is designed by Han Zhao(Keira) to implement the 
    learning algorithm for a Hidden Markov Model described in
    "Observable operator models for discrete stochastic time series, 
    Herbert Jaeger, May.10, 1999"
    '''
    def __init__(self):
        pass
    
    
    def train(self, seqlist):
        '''
        @seqlist:
            type:    list(list)
            param:   List of observation sequence

        @self.n:
            type:    numpy.uint64
            param:   Number of possible observation states
        '''
        print 'Estimating number of observation state from training data...'
        self.n = len(set([ob for seq in seqlist for ob in seq]))
        # Initialization
        self.w0 = np.zeros(self.n, dtype = np.float)
        self.V = np.zeros((self.n, self.n), dtype = np.float)
        self.W = [np.zeros((self.n, self.n), dtype = np.float) for i in xrange(self.n)]
        self.tao = [np.zeros((self.n, self.n), dtype = np.float) for i in xrange(self.n)]
        
        
        print 'Number of training list:', len(seqlist)
        # Inituition: W0, V, W learn to be the stationary distribution of Hidden Markov Model
        # If only the first three observations were used, then OOM learn to be the non-stationary
        # distribution of Hidden Markov Model
        trilist = [seq[idx: idx+3] for seq in seqlist for idx in xrange(len(seq)-2)]
        training_size = len(trilist)
        print 'Number of separated triples:', len(trilist)
        print 'Begin to counting for W0, V, W...'
        
        # Estimation by counting...
        for triple in trilist:
            self.w0[triple[0]] += 1.0
            self.V[triple[1], triple[0]] += 1.0
            self.W[triple[1]][triple[2], triple[0]] += 1.0
        
        assert np.linalg.norm(self.w0, 1) == training_size
        # Normalization
        self.w0 /= training_size
        self.V /= np.sum(self.V)
        for idx in xrange(len(self.W)):
            self.W[idx] /= np.sum(self.W[idx])
        
        inverse_v = np.linalg.inv(self.V)
        for idx in xrange(len(self.W)):
            self.tao[idx] = np.dot(self.W[idx], inverse_v)
        
        print 'Finished training...'
        for idx in xrange(len(self.W)):
            print '-' * 50
            print self.tao[idx]
        print '=' * 50
        
    
    def predict(self, seq):
        '''
        @seq:
            type:    numpy.array
            param:   Observation sequence to be predicted
        
        @return
            type:    float
            param:   The observation probability predicted by OOM model
                     P[x_t, x_t-1,...,x_1]
        '''
        prob = self.w0
        for ob in seq:
            prob = np.dot(self.tao[ob], prob)
        return np.sum(prob)
        
        
def main():
    pass


if __name__ == '__main__':
    main()
        
        
        
        
        
        
        