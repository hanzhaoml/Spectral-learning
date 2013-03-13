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
import math
import time


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
    
    def __init__(self, m = 0, n = 0, filename = None):
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
            param:   The observation matrix of HMM
        
        @sd:
            type:    numpy.array
            param:   The stationary distribution of HMM
        '''
        if filename is not None:
            self.loadModel(filename)
        else:
            assert n >= m
            
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
#        prob = math.log(np.dot(np.ones(self.m), prob))
        return prob
    
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
    
#    def beta2(self, seq):
#        t = len(seq)
#        grids = np.zeros((t, self.m), dtype = np.float)
#        grids[t-1,:] = 1.0
#        for i in xrange(t-1,0,-1):
#            for j in xrange(0, self.m):
#                for k in xrange(0, self.m):
#                    grids[i-1,j] += self.T[k,j] * self.O[seq[i],k] * grids[i,k]
#        return grids
        
        
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
       
    
def main(m, n):
    
    hmm = HMM(m, n)
    length = np.random.randint(4, 100)
    seq = [np.random.randint(0,n) for i in xrange(length)]
    prob1 = hmm.probability(seq)
    prob2 = hmm.alpha(seq)
    prob3 = hmm.beta(seq)
    print 'Probability generated by spectral learning:', prob1
    print 'Probability generated by forward algorithm:', prob2
    print 'Probability generated by backward algorithm:', prob3

if __name__ == '__main__':
    main(4, 8)

