#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.
import numpy as np
import sys
import time
import csv
import cPickle

from pprint import pprint

class HMM(object):
    '''This class defines the model of HMM (Hidden Markov Model).
    
    This class is used as a data set generator for Spectral Learning algorithm,
    which you can find at (http://arxiv.org/pdf/0811.4413v6.pdf)
    
    Given the transition matrix of HMM as T in R^(m*m), and the observation 
    matrix of HMM as O in R^(m*n), where
    
    T_{ij} = P(s'=i|s=j)
    O_{ij} = P(o=i|s=j)
    
    There are some prerequisites for T and O as follows: 
    1    rank(T) = rank(O) = m
    '''
    def __init__(self, m=0, n=0, filename=None):
        '''
        @m:    
            type:    numpy.uint64
            param:   The dimension of T, also the number of columns in T
        
        @n:    
            type:    numpy.uint64
            param:   The number of columns in O. n >= m
        
        @T:
            type:    numpy.array
            param:   The transition matrix of HMM
        
        @O:
            type:    numpy.array
            param:   The observation(emission) matrix of HMM
        
        @sd:
            type:    numpy.array
            param:   The stationary distribution of HMM
        '''
        
        pprint("In the initialize method of HMM:")
        pprint("M: %d" % m)
        pprint("N: %d" % n)
        
        self.m = m
        self.n = n
        self.T = self._randomT(m)
        self.O = self._randomO(m, n)
        (self.cT, self.cO) = self._build()
        self.sd = self._converge()
    
    def _randomT(self, m):
        '''
        @m:
            type:    numpy.uint64
            param:   The dimension of T
            
        @return:
            type:    numpy.array
            param:   Random Transition matrix of HMM 
            
        
        Given m, this procedure first builds a m-dimension matrix with random numbers.
        Then normailze each column of the matrix by one-norm (That is, 
        the sum of components in each column vector equals one)        
        '''
        while True:
            randm = np.random.rand(m, m)
            norms = np.sum(randm, axis=0)
            randm /= norms
            if np.linalg.matrix_rank(randm) == m:
                return randm
    
    def _randomO(self, m, n):
        '''
        @m:
            type:    numpy.uint64
            param:   The column dimension of O
        
        @n:
            type:    numpy.uint64
            param:   The number of rows in O
            
        @return:
            type:    numpy.array
            param:   Random Observation matrix of HMM
        '''
        while True:
            rando = np.random.rand(n, m)
            norms = np.sum(rando, axis=0)
            rando /= norms
            if np.linalg.matrix_rank(rando) == m:
                return rando

    def _build(self):
        '''
        @return:
            type:    (numpy.array, numpy.array)
            param:   Cumulative distribution of Transition matrix
                     and Cumulative distribtion of Observation matrix
                     Cumulative matrices are used to randomly generate synthetic data...
        '''
        return (np.add.accumulate(self.T, axis=0), np.add.accumulate(self.O, axis=0))        
    
    def _converge(self):
        '''
        @return:
            type:    numpy.array
            param:   stationary distribution of HMM
        '''
        (eigw, eigv) = np.linalg.eig(self.T)
        sd = eigv[:, np.abs(eigw - 1.0) < np.float(1e-6)][:, 0]
        sd = np.abs(np.real(sd) / np.linalg.norm(sd, 1))
        return sd
        
    def probability(self, seq):
        '''
        @seq:
            type:    numpy.uint64
            param:   Observation sequence
            
        @return:
            type:    numpy.float
            param:   The probability of generating this observation sequence
                     from model
        @attention:  This implementation uses the matrix operator notation to do the 
                     inference task, which may be low efficient compared with using 
                     forward-backward algorithm.
        '''
        prob = self.sd
        for ob in seq:
            diag = [self.O[ob, i] for i in xrange(self.m)]
            Ax = self.T * diag
            prob = np.dot(Ax, prob)
        prob = np.dot(np.ones(self.m), prob)
        return prob

#    alpha and beta are used for EM algorithm    
    def alpha(self, seq):
        '''
        @seq:
            type:    numpy.uint64
            param:   Observation sequence
            
        @return:    
            type:    numpy.array
            param:   P(O1,O2,...OT,ST|Model)
        '''
        t = len(seq)
        grids = np.zeros((t, self.m), dtype=np.float)
        grids[0, :] = self.sd * self.O[seq[0], :]
        for i in xrange(1, t):
            grids[i, :] = np.dot(self.T, grids[i - 1, :])
            grids[i, :] *= self.O[seq[i], :]
        return grids
    
    def beta(self, seq):
        '''
        @seq:
            type:    numpy.uint64
            param:   Observation sequence
        @return:
            type:    numpy.array
            param:    P(Ot,Ot+1,...OT|ST, Model)
        '''
        t = len(seq)
        grids = np.zeros((t, self.m), dtype=np.float)
        grids[t - 1, :] = 1.0
        for i in xrange(t - 1, 0, -1):
            grids[i - 1, :] = np.dot(grids[i, :], self.T * self.O[seq[i], :][:, np.newaxis])
        return grids
            
    def gendata(self, dsize, sqlen=100):
        '''
        @dsize:
            type:    numpy.uint64
            param:   data size to be generated
            
        @sqlen:
            type:    numpy.uint64
            param:   The maximal length of each data sequence
        
        @return:
            type:    list[list_1, list_2, ... list_dsize]
            param:   The generated observation sequence of HMM based on Monte Carlo simulation.
        '''
        data = list()
        for i in xrange(dsize):
            # Cumulative distribution of the states
            accdist = np.add.accumulate(self.sd)
            rlen = np.random.randint(3, sqlen)
            sq = np.zeros(rlen, dtype=np.uint64)
            # Initial state chosen based on the statioinary distribution
            state = (np.where(accdist >= np.random.rand())[0])[0]
            for j in xrange(rlen):
                # update the state of HMM by the Transition matrix[state]
                state = (np.where(self.cT[:, state] >= np.random.rand())[0])[0]
                # randomly choose an observation by the Observation matrix[state]
                observ = (np.where(self.cO[:, state] >= np.random.rand())[0])[0]
                sq[j] = observ
            data.append(sq)        
        return data
    
    def generate_test_data(self, seq_length):
        test_seqs = np.zeros((self.n ** seq_length, seq_length), dtype=np.int)
        return self.cartesian(seq_length, test_seqs)
    
    def generate_train_data(self, seq_length):
        train_seq = np.zeros(seq_length, dtype=np.int)
        acc_stationary_dist = np.add.accumulate(self.sd)
        # Initial state chosen based on the statioinary distribution
        state = (np.where(acc_stationary_dist >= np.random.rand())[0])[0]        
        for j in xrange(seq_length):
            # randomly choose an observation by the Observation matrix[state]
            observ = (np.where(self.cO[:, state] >= np.random.rand())[0])[0]
            # update the state of HMM by the Transition matrix[state]
            state = (np.where(self.cT[:, state] >= np.random.rand())[0])[0]
            train_seq[j] = observ
        return train_seq
    
    def cartesian(self, seq_length, out):
        '''
        @seq_length: int. Length of the observation sequence.
        @out: ndarray. All possible combinations of observation sequence of length
                        @seq_length
        '''
        dsize = self.n ** seq_length
        chunks = dsize / self.n
        out[:, 0] = np.repeat(np.arange(self.n), chunks)
        if seq_length > 1:
            self.cartesian(seq_length-1, out[0:chunks, 1:])
            for j in xrange(1, self.n):
                out[j*chunks: (j+1)*chunks, 1:] = out[0:chunks, 1:]
        return out
    
    @staticmethod
    def to_file(filename, model):
        with file(filename, "wb") as fout:
            try:
                cPickle.dump(model, fout)
                return True
            except Exception:
                return False
    
    @staticmethod
    def from_file(filename):
        with file(filename, "rb") as fin:
            try:
                model = cPickle.load(fin)
                return model
            except Exception:
                raise


class spectral_learner(object):
    '''
    Build true spectral learner based on the given HMM
    '''
    def __init__(self, hmm):
        '''
        Check two ways of computing observation probability:
        1    Pr = 1*tao*pi
        2    Pr = winf*taoprime*w0
        '''
        self.m = hmm.m
        self.n = hmm.n
        self.pi = hmm.sd
        self.tao = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
        for i in xrange(self.n):
            self.tao[i] = hmm.T * hmm.O[i, :]

#        Begining of spectral learning
#        V = O*T*diag(pi)*O^T
        self.V = np.dot(hmm.O, hmm.T) * hmm.sd
        self.V = np.dot(self.V, hmm.O.T)
        (self.U, self.S, K) = np.linalg.svd(self.V)
#        Only need first m column vectors of U
        self.U = self.U[:, 0: self.m]
        self.factor = np.dot(self.U.T, hmm.O)
        print 'True singular values of V:'
        print self.S
        print '=' * 50
        self.factor_inverse = np.linalg.inv(self.factor)
        
        self.w0 = np.dot(self.factor, hmm.sd)
        self.winf = np.dot(np.ones(self.m), np.linalg.inv(self.factor))
        self.W = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
        for i in xrange(self.n):
            self.W[i] = np.dot(self.factor, self.tao[i])
            self.W[i] = np.dot(self.W[i], self.factor_inverse)
        
    def prob_forward(self, seq):
        prob = self.pi
        for ob in seq:
            prob = np.dot(self.tao[ob], prob)
        prob = np.dot(np.ones(self.m), prob)
        return prob    
        
    def prob_operator(self, seq):
        prob = self.w0
        for ob in seq:
            prob = np.dot(self.W[ob], prob)
        prob = np.dot(self.winf, prob)
        return prob


def main(modelpath, trainset):
    t_start = time.time()
    hmm = HMM(filename=modelpath)
    operator = spectral_learner(hmm)
    t_end = time.time()
    print 'Time used to train spectral learner:', t_end - t_start
    with file(trainset, 'r') as f:
        reader = csv.reader(f)
        data = [map(int, row) for row in reader]
    print 'HMM_probability\t\tSpectral_forward\t\tSpectral_Operator'
    for seq in data:
        print hmm.probability(seq), '\t\t', operator.prob_forward(seq), '\t\t', \
                operator.prob_operator(seq), '\t\t', len(seq)
    
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'usage: modelpath trainfile'
        exit()
    main(sys.argv[1], sys.argv[2])
