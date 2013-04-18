#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.

# Lemma:
#    Given n >= 1, the probability that a matrix randomly chosen
#    from R^{n*n} is singular is 0.
#    
#    Simple proof:
#        We expand the n*n matrix into an array by row. So there are
#        n*n random variables. A n*n matrix is singular if and only if 
#        the determinant of it equals 0.
#        Denote the n*n random variables as R_{1}, R_{2}, ... R_{n*n}.
#        So the determinant of the matrix can be formed as:
#            f(R_{1}, R_{2}, ... R_{n*n})
#        where f is a multi-variant function.
#        So
#            f(R_{1}, R_{2}, ... R_{n*n}) = 0
#        is a surface in the Euclidean Space R^{n*n}. From real analysis 
#        we know that the ratio of the number of points on the surface to
#        the number of points in R^{n*n} is 0. Which means the probability 
#        of randomly generated matrix is singular is 0.
#        
#    So the algorithm in HMM used to get an invertible T and a full row rank O is 
#    highly efficient.

import numpy as np
import sys
import time
import csv

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
        if filename is not None:
            self.loadModel(filename)
        else:            
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
            norms = np.sum(randm, axis = 0)
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
            norms = np.sum(rando, axis = 0)
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
        return (np.add.accumulate(self.T, axis = 0), np.add.accumulate(self.O, axis = 0))
        
    
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
        grids = np.zeros((t, self.m), dtype = np.float)
        grids[0,:] = self.sd * self.O[seq[0],:]
        for i in xrange(1, t):
            grids[i,:] = np.dot(self.T, grids[i-1,:])
            grids[i,:] *= self.O[seq[i],:]
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
        grids = np.zeros((t, self.m), dtype = np.float)
        grids[t-1,:] = 1.0
        for i in xrange(t-1,0,-1):
            grids[i-1,:] = np.dot(grids[i,:], self.T * self.O[seq[i],:][:,np.newaxis])
        return grids
            
                
    def gendata(self, dsize, sqlen = 100):
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
            sq = np.zeros(rlen, dtype = np.uint64)
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
            
    
    def saveModel(self, filename):
        '''
        @filename:
            type:    string
            param:   file used to save the model parameters
        '''
        with file(filename, 'wb') as f:
            np.save(f, self.m)
            np.save(f, self.n)
            np.save(f, self.T)
            np.save(f, self.O)
            np.save(f, self.cT)
            np.save(f, self.cO)
            np.save(f, self.sd)
            
    
    def loadModel(self, filename):
        '''
        @filename:
            type:    string
            param:   file used to load the model parameters
        '''
        with file(filename, 'rb') as f:
            self.m = np.load(f)
            self.n = np.load(f)
            self.T = np.load(f)
            self.O = np.load(f)
            self.cT = np.load(f)
            self.cO = np.load(f)
            self.sd = np.load(f)

            
class oom_operator(object):
    '''
    Build True OOM Operator learner based on the given HMM model
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
#        tao_a = T*diag(O_a)
        self.tao = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
        for i in xrange(self.n):
            self.tao[i] = hmm.T * hmm.O[i,:]

        self.w0 = np.dot(hmm.O, hmm.sd)
#        V = O*T*diag(pi)*O'
        self.V = np.dot(hmm.O, hmm.T) * hmm.sd
        self.V = np.dot(self.V, hmm.O.T)
        self.V_inverse = np.linalg.inv(self.V)
#        print 'Rank of V:', np.linalg.matrix_rank(self.V)
#        Wa = O*Tao_a*T*diag(pi)*O'
        self.W = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
        for i in xrange(self.n):
            self.W[i] = np.dot(hmm.O, self.tao[i])
            self.W[i] = np.dot(self.W[i], hmm.T) * hmm.sd
            self.W[i] = np.dot(self.W[i], hmm.O.T)
        self.taoprime = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
        for i in xrange(self.n):
            self.taoprime[i] = np.dot(self.W[i], self.V_inverse)
            self.taoprime[i][np.abs(self.taoprime[i]) < 1e-16] = 0.0
#        self.winf = np.dot(self.w0.T, self.V_inverse)
#        just another trial, it should be right now
        self.winf = np.dot(np.ones(self.m), np.linalg.pinv(hmm.O))
        
        
    def print_out(self, hmm):
        '''
        Print out parameters of current oom_operator
        '''
        print 'Parameters in Hidden Markov Model'
        print 'Transition matrix:'
        print hmm.T
        print '-' * 50
        print 'Emission matrix:'
        print hmm.O
        print '-' * 50
        print 'Initial distribution:'
        print hmm.sd
        print '=' * 50
        
        print 'Parameters in Observable Operator Model'
        print 'V matrix:'
        print self.V
        print '-' * 50
        print 'V inverse matrix:'
        print self.V_inverse
        print '-' * 50
        print 'V*V^-1 matrix:'
        print np.dot(self.V, self.V_inverse)
        print '-' * 50
        print 'V^-1*V matrix:'
        print np.dot(self.V_inverse, self.V)
        print '=' * 50
        print 'Tao_a = T*diag(O_a) matrix:'
        for matrix in self.tao:
            print matrix
            print '-' * 50
        print '=' * 50
        print 'W_a = OTao_a Tdiag(pi)*O^T'
        for matrix in self.W:
            print matrix
            print '-' * 50
        print '=' * 50
    
    def prob_forward(self, seq):
        prob = self.pi
        for ob in seq:
            prob = np.dot(self.tao[ob], prob)
        prob = np.dot(np.ones(self.m), prob)
        return prob

    
    def prob_operator(self, seq):
        prob = self.w0
        for ob in seq:
            prob = np.dot(self.taoprime[ob], prob)
        prob = np.dot(self.winf, prob)
        return prob
    

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
#        tao_a = T*diag(O_a)
        self.tao = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]
        for i in xrange(self.n):
            self.tao[i] = hmm.T * hmm.O[i,:]

#        Begining of spectral learning
#        V = O*T*diag(pi)*O^T
        self.V = np.dot(hmm.O, hmm.T) * hmm.sd
        self.V = np.dot(self.V, hmm.O.T)
        (self.U, S, K) = np.linalg.svd(self.V)
#        Only need first m column vectors of U
        self.U = self.U[:, 0: self.m]
        self.factor = np.dot(self.U.T, hmm.O)
        
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

        
    def print_out(self):
        pass
    
    
    
def main(modelpath, trainset):
    hmm = HMM(filename=modelpath)
    t_start = time.time()
    operator = spectral_learner(hmm)
    t_end = time.time()
    print 'Time used to train spectral learner:', t_end-t_start
#    operator.print_out(hmm)
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


