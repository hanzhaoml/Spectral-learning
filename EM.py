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
    def __init__(self, num_hidden, num_observ):
        '''
        @num_hideen: np.int, number of hidden states in HMM.
        @num_observ: np.int, number of observations in HMM.
        '''
        self._num_hidden = num_hidden
        self._num_observ = num_observ
#         Randomly generate the transition matrix from uniform [0, 1] and normalize it
        self._transition_matrix = np.random.rand(self._num_hidden, self._num_hidden)    
        norms = np.sum(self._transition_matrix, axis=0)
        self._transition_matrix /= norms
#         Randomly generate the observation matrix and then again, normalize it
        self._observation_matrix = np.random.rand(self._num_observ, self._num_hidden)
        norms = np.sum(self._observeration_matrix, axis=0)
        self._observation_matrix /= norms
#         Randomly generate the initial distribution pi
        self._pi = np.random.rand(self._num_hidden)
        norms = np.sum(self._pi)
        self._pi /= norms
    
    def predict(self, seq):
        return np.sum(self._alpha(seq)[len(seq) - 1, :])
    
    @property
    def transition_matrix(self):
        return self._transition_matrix
    
    @property
    def observation_matrix(self):
        return self._observation_matrix
        
    @property
    def initial_dist(self):
        return self._pi
    
    def _alpha(self, seq):
        '''
        @seq:    np.ndarray, observation sequence
        @attention: Inside function used to compute the forward probability in HMM
                    using Dynamic Programming.
        '''
        t = len(seq)
        grids = np.zeros((t, self._num_hidden), dtype=np.float)
        grids[0, :] = self._pi * self._observation_matrix[seq[0], :]
        for i in xrange(1, t):
            grids[i, :] = np.dot(self._transition_matrix, grids[i - 1, :])
            grids[i, :] *= self._observation_matrix[seq[i], :]
        return grids
    
    def _beta(self, seq):
        '''
        @seq:    np.ndarray, observation sequence
        @attention: Inside function used to compute the backward probability in HMM
                    using Dynamic Programming.
        '''
        t = len(seq)
        grids = np.zeros((t, self._num_hidden), dtype=np.float)
        grids[t - 1, :] = 1.0
        for i in xrange(t - 1, 0, -1):
            grids[i - 1, :] = np.dot(grids[i, :],
            self._transition_matrix * self._observation_matrix[seq[i], :][:, np.newaxis])
        return grids

    def train(self, sequence, seg_length=100):
        '''
        @sequence: np.ndarray. A long sequence of observations.
        @attention: This version of EM algorithm has been vectorized, hence being more efficient
        '''
#         When train is called each time, start with different random points
        self._transition_matrix = np.random.rand(self._num_hidden, self._num_hidden)
        norms = np.sum(self._transition_matrix, axis=0)
        self._transition_matrix /= norms
        
        self._observation_matrix = np.random.rand(self._num_observ, self._num_hidden)
        norms = np.sum(self._observation_matrix, axis=0)
        self._observation_matrix /= norms
        
        self._pi = np.random.rand(self._num_hidden)
        norms = np.sum(self._pi)
        self._pi /= norms
        # Partition the sequence first to avoid numeric issues
        num_partitions = len(sequence) / seg_length
        seq_lists = [sequence[i * seg_length: (i + 1) * seg_length] for i in xrange(num_partitions)]
        if len(sequence) > num_partitions * seg_length:
            seq_lists.append(sequence[num_partitions * seg_length:])            
        seq_lists = np.array(seq_lists)
        threshold = self._num_hidden / 500.0
        iters = 0
        # Forward-Backward Algorithm to train HMM on seq_lists
        while True:
            iters += 1
            pi = np.zeros(self._num_hidden, dtype=np.float)
            transition = np.zeros((self._num_hidden, self._num_hidden), dtype=np.float)
            observation = np.zeros((self._num_observ, self._num_hidden), dtype=np.float)
            for seq in seq_lists:
                # Forward-probability
                alpha_matrix = self._alpha(seq)
                # Backward-probability
                beta_matrix = self._beta(seq)
                # Constructing sigma_matrix
                sigma_matrix = self._transition_matrix * np.outer(self._observation_matrix[seq[1], :] * beta_matrix[1, :], alpha_matrix[0, :])
                sigma_matrix /= np.sum(sigma_matrix)
                # Updating pi
                pi += np.sum(sigma_matrix, axis=0)
                for k in xrange(len(seq) - 1):
                    sigma_matrix = self._transition_matrix * np.outer(self._observation_matrix[seq[k + 1], :] * beta_matrix[k + 1, :], alpha_matrix[k, :])
                    sigma_matrix /= np.sum(sigma_matrix)
                    transition += sigma_matrix
                    observation[seq[k], :] += np.sum(sigma_matrix, axis=0)
            transition /= np.sum(transition, axis=0)[np.newaxis, :]
            observation /= np.sum(observation, axis=0)[np.newaxis, :]
            pi /= np.sum(pi)
            if np.sum(np.abs(self._transition_matrix - transition)) < threshold and \
                np.sum(np.abs(self._observation_matrix - observation)) < threshold:
                break
            self._transition_matrix = transition
            self._observation_matrix = observation
            self._pi = pi
        # local_max_obj = np.sum(np.log([self.predict(seq) for seq in seq_lists]))
        
