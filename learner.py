#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.

import numpy as np
import math

class SpectralLearner(object):
    '''This class implements the spectral learning algorithm described in:
            A spectral algorithm for learning hidden markov models
            http://arxiv.org/pdf/0811.4413v6.pdf
    '''
    
    def __init__(self):
        self.threshold = 1e-6
    
    def train(self, trilst):
        '''
        @trilst:
            type:    numpy.array, shape(N, 3) where N is the size of triples.
            param:   Training set
            
        @self.n:
            type:    numpy.uint64
            param:   Number of possible observation states
             
        @self.P_1:
            type:    numpy.array, shape(self.n)
            param:   Estimation of P[x_1]
        
        @self.P_21:
            type:    numpy.array, shape(self.n, self.n)
            param:   Estimation of P[x_2, x_1]
            
        @self.P_3x1:
            type:    list[numpy.array, shape(self.n, self.n)]
            param:   Estimation of P[x_3, x, x_1]

        @self.S:
            type:    numpy.array
            param:   Singular value of P_21
            
        @self.m:
            type:    numpy.uint64
            param:   len(self.S), also the number of pseudo states
            
        @self.U:
            type:    numpy.array, shape (self.n, self.m)
            param:   Left singular matrix of P_21
            
        @self.b1:
            type:    numpy.array, shape (m, 1)
            param:   analog initial vector
        
        @self.binf:
            type:    numpy.array, shape (m, 1)
            param:   analog infinity vector
        
        @self.Bx:
            type:    numpy.array, shape (m, m)
            param:   analog of the Transition matrix
        
        
        Frequency based estimation of P_1, P_21 and P_3x1, where x = 1, 2, ... n, n is the
        size of observation states.
        '''
        print 'Estimating n from observable data...'
        self.n = len(set([ob for sublist in trilst for ob in sublist]))
        
        self.P_1 = np.zeros(self.n, dtype=np.float)
        self.P_21 = np.zeros((self.n, self.n), dtype=np.float)
        self.P_3x1 = [np.zeros((self.n, self.n), dtype=np.float) for i in xrange(self.n)]

        print 'Number of input sequences:', len(trilst)
        trilst = [sublst[idx: idx + 3] for sublst in trilst for idx in range(len(sublst) - 2)]
        print 'Number of separated triples:', len(trilst)
        print 'Estimate P_1, P_21 and P_3x1...'
        # Parameter estimation
        # Frequency based estimation
        for sq in trilst:
            self.P_1[sq[0]] += 1.0
            self.P_21[sq[1], sq[0]] += 1.0
            self.P_3x1[sq[1]][sq[2], sq[0]] += 1.0
        
#        self.checkSparse()

        print 'Normalizing P_1, P_21 and P_3x1...'
        # Normalization of P_1, P_21, P_3x1
        norms = np.linalg.norm(self.P_1, 1)
        self.P_1 /= norms

        # Normalize the joint distribution of P_21        
        norms = np.sum(self.P_21)
        self.P_21 /= norms
        
        # Normalize the joint distribution of P_3x1
        for i in xrange(len(self.P_3x1)):
            norms = np.sum(self.P_3x1[i])
            self.P_3x1[i] /= norms

        # Singular Value Decomposition
        # Keep all the positive singular values
        (U, S, V) = np.linalg.svd(self.P_21)
        # TODO(Keira):    Decide whether it should discard very small
        #                 singular values
#        self.S = S[S > self.threshold]
        self.S = S;
        
        self.m = self.S.shape[0]
        self.U = U[:, 0:self.m]
        self.V = V[0:self.m, :]
        
        print '=' * 50        
        print 'Left singular matrix of P_21 SVD:'
        print self.U
        print '*' * 50
        print 'Singular values of P_21 SVD:'
        print self.S
        print '*' * 50
        print 'Transpose of right singular matrix of P_21 SVD:'
        print self.V
        print '=' * 50
        
#        # Checking whether U is an orthogonal matrix or not
#        print 'U^T * U:'
#        print np.dot(self.U.T, self.U)
#        print '-' * 50
#        # Checking whether V^T is an orthogonal matrix or not
#        print 'V^T * V:'
#        print np.dot(self.V, self.V.T)
#        print '-' * 50
                
        # TODO(Keira)    U should be normalized by column vector of norm 2
        #                but not row vector of norm 1. U is the left singular
        #                matrix of P_21
#        norms = np.sum(self.U, axis=1)
#        self.U /= norms[:, np.newaxis]
        
#        print 'm:', self.m  
#        print '-' * 50
#        print 'U matrix:', self.U
        
        print 'Constructing b_1, b_inf and B_x...'
        # Compute b1, binf and Bx
        # self.factor = (P_21^{T} * U)^{+}, which is used to accelerate the computation
        self.factor = np.linalg.pinv(np.dot(self.P_21.T, self.U))
        norms = np.sum(self.factor, axis=1)
                
        self.b1 = np.dot(self.U.T, self.P_1)
        
        s = np.sum(self.b1)
        print 'sum of b1 vector:', s
        print 'b1 vector:'
        print self.b1
        
        self.binf = np.dot(self.factor, self.P_1)
        s = np.sum(self.binf)
        print 'sum of binf vector:', s
        print 'binf vector:'
        print self.binf
        print '-' * 50
        
        
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
            param:   The log_e probability of the observation sequence estimated by 
                     the learned model
                    
                     P[x_t+1, x_t, x_t-1, ... x1]
        '''
        
        # binf * B_{xt:1} * b1
        prob = self.b1

        for ob in seq:
            prob = np.dot(self.Bx[ob], prob)
        prob = np.dot(self.binf.T, prob)
        for i in xrange(len(seq)):
            prob /= self.n
        return prob
        
    def checkSparse(self):
        '''
        Output the sparsity of P_1, P_21 and P_3x1
        '''
        zero_cnt = len(self.P_1[self.P_1 == 0])
        print '0 count:', zero_cnt
        print '0 ratio in self.P_1:', float(zero_cnt) / len(self.P_1)
        
        print '-' * 50
        
        zero_cnt = len(self.P_21[self.P_21 == 0])
        print '0 count:', zero_cnt
        print '0 ratio in self.P_21:', float(zero_cnt) / len(self.P_21.reshape(-1))
        
        print 'P_1:'
        print self.P_1
        print '-' * 50
        
        print 'P_21:'
        print self.P_21
        print '-' * 50
        
        print 'P_3x1:'
        print self.P_3x1
        print '-' * 50
        
        
def main():
    pass



if __name__ == '__main__':
    main()

        
        
        
        
        
        
        
