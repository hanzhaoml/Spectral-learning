#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.



import numpy as np


class BaumWelch(object):
    '''
    This class is used to learning HMM using Baum-Welch algorithm,
    which is an implementation of Expectation-Maximization algorithm.
    '''
    def __init__(self, m, n):
        self.m = m
        self.n = n
    
        self.T = np.random.rand(m, m)
        norms = np.sum(self.T, axis = 0)
        self.T /= norms
        
        self.O = np.random.rand(n, m)
        norms = np.sum(self.O, axis = 0)
        self.O /= norms
        
        self.sd = np.random.rand(m)
        norms = np.sum(self.sd)
        self.sd /= norms
    
    def predict(self, seq):
        return np.sum(self.alpha(seq)[len(seq)-1,:])
        
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


    def train(self, trainset):
        '''
        Baum-Welch algorithm, also known as the Forward-Backward Algorithm.
        The Baum-Welch algorithm lies under the framework of EM algorithm.
        '''
        threshold = self.m / 1000.0
        iters = 0
        while True:
#            gamma_1 is used to update for initial distribution, accumulating over all observations
#            sigma is used to update for transition matrix, accumulating over all observations
#            gammaa is used to update for observation matrix, accumulating over all observations
            iters += 1
            gamma_1 = np.zeros(self.m, dtype = np.float)
            sigma = np.zeros((self.m, self.m), dtype = np.float)
            gamma = np.zeros((self.n, self.m), dtype = np.float)
#            For each observation sequence, update the frequency statistics
            for seq in trainset:
                l = len(seq)
#                temporal estimator of alpha, beta, sigma_t, gamma_t
                alpha = self.alpha(seq)
                beta = self.beta(seq)
                
                sigma_t = np.zeros((self.m, self.m), dtype = np.float)
                for i in xrange(self.m):
                    for j in xrange(self.m):
                        sigma_t[j,i] = alpha[0,i] * self.T[j,i] * self.O[seq[1],j] * beta[1,j]
                sigma_t /= np.sum(sigma_t)
#                update statistics about initial distribution statistics
                gamma_1 += np.sum(sigma_t, axis = 0)
#                assert: for each sequence, np.sum(delta sigma) = 1 and np.sum(delta gamma) = 1
                for k in xrange(l-1):
                    for i in xrange(self.m):
                        for j in xrange(self.m):
                            sigma_t[j,i] = alpha[k,i] * self.T[j,i] * self.O[seq[k+1],j] * beta[k+1,j]
                    sigma_t /= np.sum(sigma_t)
                    sigma += sigma_t
                    gamma[seq[k],:] += np.sum(sigma_t, axis = 0)
#            assert: np.sum(sigma) == len(seq), np.sum(gamma) == len(seq)
            sigma /= np.sum(sigma, axis = 0)[np.newaxis,:]
            gamma /= np.sum(gamma, axis = 0)[np.newaxis,:]
            print 'End of iterations', iters
            if np.sum(np.abs(self.T - sigma)) < threshold and np.sum(np.abs(self.O - gamma)) < threshold:
                break
            self.T = sigma
            self.O = gamma
        
                
#        for seq in trainset:
#            print '-' * 50
#            if len(set(seq[:-1])) < self.n:
#                continue
#            print 'length of sequence:', len(seq)
#            print 'Observation sequence:', seq
#            print 'Current self.T:'
#            print self.T
#            print 'Current self.O:'
#            print self.O
#            count += 1
#            sigma = np.zeros((self.m, self.m), dtype = np.float)
#            gamma = np.zeros((self.n, self.m), dtype = np.float)
#            t = len(seq)
##            print '%f%%' %(count * 100 / sz)
#            
#            alpha = self.alpha(seq)
#            beta = self.beta(seq)
#            
##            Re-estimate of stationary distribution
#            for i in xrange(self.m):
#                for j in xrange(self.m):
#                    sigma[j,i] = alpha[0,i] * self.T[j,i] * self.O[seq[1],j] * beta[1, j]
#            ssum = np.sum(sigma)
##            print 'sigma:'
##            print sigma
##            print 'sum of stationary distribution', np.sum(sigma)
##            print 'Alpha:'
##            print alpha
##            print 'Beta:'
##            print beta
#            
#            sigma /= ssum 
#            self.sd = np.sum(sigma, axis = 0)
#            self.sd /= np.sum(self.sd)
#            sigma = np.zeros((self.m, self.m), dtype = np.float)
###            Re-estimate of T and O
#            for l in xrange(0, t-1):
#                sigma_t = np.zeros((self.m, self.m), dtype = np.float)
#                gamma_t = np.zeros((self.m, self.m), dtype = np.float)
#                for i in xrange(self.m):
#                    for j in xrange(self.m):
#                        sigma_t[j,i] = alpha[l,i] * self.T[j,i] * self.O[seq[l+1],j] * beta[l+1,j]
#                        gamma_t[seq[l],i] += sigma_t[j, i]
##                print 'l = ', l
##                print 'sigma_t:'
##                print sigma_t
##                print 'gamma_t:'
##                print gamma_t
##                print 'sum of sigma_t:', np.sum(sigma_t, axis = 0)
##                print 'sum sigma_t:', np.sum(sigma_t)
##                print '-' * 50
#                gamma_t /= np.sum(sigma_t)
#                sigma_t /= np.sum(sigma_t)
#                gamma += gamma_t
#                sigma += sigma_t
##            print '#' * 50
##            print 'np.sum(gamma):', np.sum(gamma, axis = 0)[np.newaxis,:]
##            print 'Gamma for a sequence:'
##            print gamma
##            print 'Sigma for a sequence:'
##            print sigma
##            print '*' * 50
#            self.T = sigma / np.sum(gamma, axis = 0)[np.newaxis,:]
#            self.O = gamma / np.sum(gamma, axis = 0)[np.newaxis,:]
##            print 'Transition matrix:'
##            print self.T
##            print '-' * 50
##            print 'Emission matrix:'
##            print self.O
##            print '=' * 75
##       
        
        
        
        
def main():
    pass


if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        
        
        