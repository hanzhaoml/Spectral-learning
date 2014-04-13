#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) Apr 12, 2014 Han Zhao <han.zhao@uwaterloo.ca>
'''
@purpose: Base class for Hidden Markov Model
@author: keira
'''
import numpy as np


class HMM(object):
    '''
    Hidden Markov Model (HMM) for multinomial observation sequence.
    
    Provide basic interface to be implemented by different training algorithms, 
    including but not limited to Expectation Maximization, Spectral Learning, etc.
    In general, HMM can be used to solve three kinds of problems:
    1,     Estimation problem:
           Given an observable sequence, o1, o2, ..., ot, computing the marginal 
           probability Pr(o1, o2, ..., ot).
    
    2,     Decoding problem:
           Given an observable sequence, o1, o2, ..., ot, infer the hidden state sequence
           s1, s2, ..., st which gives the largest probability for o1, o2, ..., ot, 
           i.e., decoding problem solves the following problem:
           s1, s2, ..., st = argmax Pr(o1, o2, ..., ot|s1, s2, ..., st)
    
    3,     Learning problem:
           Given a set of observation sequences, infer for the transitioin matrix, observation
           matrix and initial distribution, i.e., learning problem solves the following problem:
           T, O, pi = argmax Pr(X | T, O, pi)
           where X is the set of observation sequences.
    '''
    def __init__(self, num_hidden, num_observ,
                 transition_matrix=None, observation_matrix=None, initial_dist=None):
        '''
        @num_hidden: np.int. Number of hidden states in the HMM.
        @num_observ: np.int. Number of observations in the HMM. 
        
        @transition_matrix: np.ndarray, shape = (num_hidden, num_hidden)
                            Transition matrix of the HMM, denoted by T, 
                            T_ij = Pr(h_t+1 = i | h_t = j)
                            
                            Default value is None. If it is not None, it must satisfy
                            the following two conditions:
                            1.     shape = (num_hidden, num_hidden)
                            2.     All the elements should be non-negative
                            
                            Note: The input transition matrix will be normalized column-wisely
        @observation_matrix: np.ndarray, shape = (num_observ, num_hidden)
                             Observation matrix of the HMM, denoted by O,
                             O_ij = Pr(o_t = i | h_t = j)
                            
                            Default value is None. If it is not None, it must satisfy 
                            the following two conditions:
                            1.        shape = (num_observ, num_hidden)
                            2.        All the elements should be non-negative
                            
                            Note: The input observation matrix will be normalized column-wisely
        @initial_dist: np.ndarray, shape = (num_hidden,)
                       Initial distribution for hidden states.
                       Pi_i = Pr(h_1 = i)
                       
                       Default value is None. If it is not None, it must satisfy the following two
                       conditions:
                       1.     shape = (num_hidden,)
                       2.     All the elements should be non-negative
                       
                       Note: The input array will be normalized to form a probability distribution.
        '''
        if num_hidden <= 0 or not isinstance(num_hidden, int):
            raise ValueError("Number of hidden states should be positive integer")
        if num_observ <= 0 or not isinstance(num_observ, int):
            raise ValueError("Number of observations should be positive integer")
        self._num_hidden = num_hidden
        self._num_observ = num_observ
        # Build transition matrix, default is Identity matrix 
        if transition_matrix != None:
            if not (transition_matrix.shape == (num_hidden, num_hidden)):
                raise ValueError("Transition matrix should have size: (%d, %d)" 
                                 % (num_hidden, num_hidden))
            if not np.all(transition_matrix >= 0):
                raise ValueError("Elements in transition matrix should be non-negative")
            self._transition_matrix = transition_matrix
        else:
            self._transition_matrix = np.eye(num_hidden, dtype=np.float)
        # Build observation matrix, default is Identity matrix
        if observation_matrix != None:
            if not (observation_matrix.shape == (num_observ, num_hidden)):
                raise ValueError("Observation matrix should have size: (%d, %d"
                                 % (num_observ, num_hidden))
            if not np.all(observation_matrix >= 0):
                raise ValueError("Elements in observation matrix should be non-negative")
            self._observation_matrix = observation_matrix
        else:
            self._observation_matrix = np.eye(num_observ, num_hidden)
        # Build initial distribution, default is uniform distribution
        if initial_dist != None:
            if not (initial_dist.shape[0] == num_hidden):
                raise ValueError("Initial distribution should have length: %d" % num_hidden)
            if not np.all(initial_dist >= 0):
                raise ValueError("Elements in initial_distribution should be non-negative")
            self._initial_dist = initial_dist
        else:
            self._initial_dist = np.ones(num_hidden, dtype=np.float)
            self._initial_dist /= num_hidden
    
    ##############################################################################
    # Public methods
    ##############################################################################        
    @property
    def initial_dist(self):
        return self._initial_dist
    
    @property
    def transition_matrix(self):
        return self._transition_matrix
    
    @property
    def observation_matrix(self):
        return self._observation_matrix

    def decode(self, sequence):
        '''
        Solve the decoding problem with HMM, also called "Viterbi Algorithm"
        @sequence: np.array. Observation sequence
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ)
        @note: Using dynamic programming to find the most probable hidden state
               sequence, the computational complexity for this algorithm is O(Tm^2),
               where T is the length of the sequence and m is the number of hidden
               states.
        '''
        t = len(sequence)
        prob_grids = np.zeros((t, self._num_hidden), dtype=np.float)
        path_grids = np.zeros((t, self._num_hidden), dtype=np.int)
        # Boundary case
        prob_grids[0, :] = self._initial_dist * self._observation_matrix[sequence[0], :]
        path_grids[0, :] = -1
        # DP procedure, prob_grids[i, j] = max{prob_grids[i-1, k] * T_{j,k} * O_{seq[i],j}}
        # Forward-computing of DP procedure
        for i in xrange(1, t):
            # Using vectorized code to avoid the explicit for loop, improve the efficiency,
            # i.e., H_k(i) = max{ H_{k-1}(j) * T_{i,j} * O_{seq[k], i}}, which can be formed
            # as a matrix by using outer product
            exp_prob = np.outer(self._observation_matrix[sequence[i], :], prob_grids[i-1, :])
            exp_prob *= self._transition_matrix
            prob_grids[i, :], path_grids[i, :] = \
            np.max(exp_prob, axis=1), np.argmax(exp_prob, axis=1)
        # Backward-path finding of DP procedure
        opt_hidden_seq = np.zeros(t, dtype=np.int)
        opt_hidden_seq[-1] = np.argmax(prob_grids[-1, :])
        for i in xrange(t-1, 0, -1):
            opt_hidden_seq[i-1] = path_grids[i, opt_hidden_seq[i]]
        return opt_hidden_seq
    
    def predict(self, sequence):
        '''
        Solve the estimation problem with HMM.
        @sequence: np.array. Observation sequence
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ)
        '''
        return np.sum(self._alpha_process(sequence)[-1, :])

    def fit(self, sequences):
        '''
        Solve the learning problem with HMM.
        @sequences: [np.array]. List of observation sequences, each observation 
                    sequence can have different length
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ)
        @note: This method should be overwritten by different subclass. 
        '''
        pass
    
    #########################################################################
    # Protected methods
    #########################################################################
    def _alpha_process(self, sequence):
        '''
        @sequence: np.array. Observation sequence
        @note: Computing the forward-probability: Pr(o_1,o_2,...,o_t, h_t=i) 
               using dynamic programming. The computational complexity is O(Tm^2),
               where T is the length of the observation sequence and m is the 
               number of hidden states in the HMM.
        '''
        t = len(sequence)
        grids = np.zeros((t, self._num_hidden), dtype=np.float)
        grids[0, :] = self._initial_dist * self._observation_matrix[sequence[0], :]
        for i in xrange(1, t):
            grids[i, :] = np.dot(self._transition_matrix, grids[i-1, :])
            grids[i, :] *= self._observation_matrix[sequence[i], :]
        return grids
    
    def _beta_process(self, sequence):
        '''
        @sequence: np.array. Observation sequence
        @note: Computing the backward-probability: Pr(o_t+1, ..., o_T | h_t)
               using dynamic programming. The computational complexity is 
               O(Tm^2) where T is the length of the observation sequence and m
               is the number of hidden states in the HMM.
        '''
        t = len(sequence)
        grids = np.zeros((t, self._num_hidden), dtype=np.float)
        grids[t-1, :] = 1.0
        for i in xrange(t-1, 0, -1):
            grids[i-1, :] = grids[i, :] * self._observation_matrix[sequence[i], :]
            grids[i-1, :] = np.dot(grids[i-1, :], self._transition_matrix)
        return grids
