#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.

from pprint import pprint
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
        norms = np.sum(self.T, axis=0)
        self.T /= norms
        
        self.O = np.random.rand(n, m)
        norms = np.sum(self.O, axis=0)
        self.O /= norms
        
        self.sd = np.random.rand(m)
        norms = np.sum(self.sd)
        self.sd /= norms
    
    def predict(self, seq):
        return np.sum(self.alpha(seq)[len(seq)-1, :])
    
    @property
    def transition_matrix(self):
        return self.T
    
    @property
    def observation_matrix(self):
        return self.O
        
    @property
    def stationary_dist(self):
        return self.sd
    
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
            grids[i, :] = np.dot(self.T, grids[i-1, :])
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
        grids[t-1, :] = 1.0
        for i in xrange(t-1, 0, -1):
            grids[i-1, :] = np.dot(grids[i, :], self.T * self.O[seq[i], :][:, np.newaxis])
        return grids

    def train(self, sequence, seg_length=100):
        '''
        @sequence: ndarray. A long sequence of observations.
        
        @attention: This version of EM algorithm has been vectorized, hence being more efficient
        '''
        # Partition the sequence first to avoid numeric issues
        num_partitions = len(sequence) / seg_length
        seq_lists = [sequence[i*seg_length: (i+1)*seg_length] for i in xrange(num_partitions)]
        if len(sequence) > num_partitions * seg_length:
            seq_lists.append(sequence[num_partitions*seg_length:])            
        seq_lists = np.array(seq_lists)
        threshold = self.m / 500.0
        iters = 0
        # Forward-Backward Algorithm to train HMM on seq_lists
        while True:
            iters += 1
            pi = np.zeros(self.m, dtype=np.float)
            transition = np.zeros((self.m, self.m), dtype=np.float)
            observation = np.zeros((self.n, self.m), dtype=np.float)
            for seq in seq_lists:
                # Forward-probability
                alpha_matrix = self.alpha(seq)
                # Backward-probability
                beta_matrix = self.beta(seq)
                # Constructing sigma_matrix
                sigma_matrix = self.T * np.outer(self.O[seq[1], :] * beta_matrix[1, :], alpha_matrix[0, :])
                sigma_matrix /= np.sum(sigma_matrix)
                # Updating pi
                pi += np.sum(sigma_matrix, axis=0)
                for k in xrange(len(seq)-1):
                    sigma_matrix = self.T * np.outer(self.O[seq[k+1], :] * beta_matrix[k+1, :], alpha_matrix[k, :])
                    sigma_matrix /= np.sum(sigma_matrix)
                    transition += sigma_matrix
                    observation[seq[k], :] += np.sum(sigma_matrix, axis=0)
            transition /= np.sum(transition, axis=0)[np.newaxis, :]
            observation /= np.sum(observation, axis=0)[np.newaxis, :]
            pi /= np.sum(pi)
            if np.sum(np.abs(self.T - transition)) < threshold and np.sum(np.abs(self.O - observation)) < threshold:
                break
            self.T = transition
            self.O = observation
            self.sd = pi


def main():
    pass


if __name__ == '__main__':
    main()
