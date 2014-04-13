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
from pprint import pprint

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


class EMHMM(HMM):
    '''
    This class is used to learning HMM using Baum-Welch algorithm, also
    called Forward-Backward algorithm, which is an implementation of 
    Expectation-Maximization algorithm.
    '''
    def __init__(self, num_hidden, num_observ, 
                 transition_matrix=None, observation_matrix=None, init_dist=None):
        '''
        @num_hideen: np.int, number of hidden states in HMM.
        @num_observ: np.int, number of observations in HMM.
        
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
        # Call initial method of base class directly
        super(EMHMM, self).__init__(num_hidden, num_observ, 
                       transition_matrix, observation_matrix, init_dist)
    
    # Override the fit algorithm provided in HMM
    def fit(self, sequences, max_iters=500, repeats=20, seq_length=100,
            threshold=None, verbose=False):
        '''
        Solve the learning problem with HMM.
        @sequences: [np.array]. List of observation sequences, each observation 
                    sequence can have different length
        @repeats (optional): np.int. Rerun the Expectation-Maximization algorithm
                            @repeats times and choose the parameter setting which
                            gives the highest training-set log-likelihood
        
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ). Note that each sequence in @sequences
                    should not be too long, or there will be numerical issues, like 
                    underflow. Typical length of each observation sequence should be 
                    less than around 300. 
        '''
        # Set default threshold
        if threshold == None: threshold = self._num_hidden / 500.0
        # Partition long sequence first to avoid numeric issues
        short_sequences = filter(lambda seq: len(seq) <= seq_length, sequences)
        long_sequences = filter(lambda seq: len(seq) > seq_length, sequences)
        for each_sequence in long_sequences:
            num_partitions = len(each_sequence) / seq_length
            seq_segments = [each_sequence[j * seq_length: (j + 1) * seq_length] 
                            for j in xrange(num_partitions)]
            if len(each_sequence) > num_partitions * seq_length:
                seq_segments.append(each_sequence[num_partitions * seq_length:])            
            short_sequences.extend(seq_segments)
        if verbose: 
            pprint("Total number of reduced short sequences: %d" % len(short_sequences))
        # Variables used to store optimal parameters found in iterations
        opt_transition_matrix = np.random.rand(self._num_hidden, self._num_hidden)
        opt_observation_matrix = np.random.rand(self._num_observ, self._num_hidden)
        opt_init_dist = np.random.rand(self._num_hidden)
        opt_log_likelihoods = -np.inf
        for i in xrange(repeats):
            #When train is called each time, start with different random points
            self._transition_matrix = np.random.rand(self._num_hidden, self._num_hidden)
            norms = np.sum(self._transition_matrix, axis=0)
            self._transition_matrix /= norms
            
            self._observation_matrix = np.random.rand(self._num_observ, self._num_hidden)
            norms = np.sum(self._observation_matrix, axis=0)
            self._observation_matrix /= norms
            
            self._init_dist = np.random.rand(self._num_hidden)
            norms = np.sum(self._init_dist)
            self._init_dist /= norms
                
            iters = 0
            # Forward-Backward Algorithm to train HMM on seq_lists
            while iters < max_iters:
                iters += 1
                pi = np.zeros(self._num_hidden, dtype=np.float)
                transition = np.zeros((self._num_hidden, self._num_hidden), dtype=np.float)
                observation = np.zeros((self._num_observ, self._num_hidden), dtype=np.float)
                for seq in short_sequences:
                    # Forward-probability
                    alpha_matrix = self._alpha_process(seq)
                    # Backward-probability
                    beta_matrix = self._beta_process(seq)
                    # Constructing sigma_matrix
                    sigma_matrix = self._transition_matrix * \
                    np.outer(self._observation_matrix[seq[1], :] * beta_matrix[1, :], alpha_matrix[0, :])
                    sigma_matrix /= np.sum(sigma_matrix)
                    # Updating pi
                    pi += np.sum(sigma_matrix, axis=0)
                    pi /= np.sum(pi)
                    for k in xrange(len(seq) - 1):
                        sigma_matrix = self._transition_matrix * \
                        np.outer(self._observation_matrix[seq[k + 1], :] * beta_matrix[k + 1, :], alpha_matrix[k, :])
                        sigma_matrix /= np.sum(sigma_matrix)
                        transition += sigma_matrix
                        observation[seq[k], :] += np.sum(sigma_matrix, axis=0)
                # Updating transition and observation matrix
                transition /= np.sum(transition, axis=0)[np.newaxis, :]
                observation /= np.sum(observation, axis=0)[np.newaxis, :]
                if np.sum(np.abs(self._transition_matrix - transition)) < threshold and \
                    np.sum(np.abs(self._observation_matrix - observation)) < threshold:
                    break
                self._transition_matrix = transition
                self._observation_matrix = observation
                self._pi = pi
            log_likelihoods = np.sum(np.log([self.predict(seq) for seq in short_sequences]))
            if log_likelihoods > opt_log_likelihoods:
                opt_log_likelihoods = log_likelihoods
                opt_transition_matrix = self._transition_matrix
                opt_observation_matrix = self._observation_matrix
                opt_init_dist = self._initial_dist
        # Setting parameter which gains the highest log-likelihoods
        self._transition_matrix = opt_transition_matrix
        self._observation_matrix = opt_observation_matrix
        self._initial_dist = opt_init_dist
