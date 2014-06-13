#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) Apr 12, 2014 Han Zhao <han.zhao@uwaterloo.ca>
'''
@purpose: Base class for Hidden Markov Model
@author: Han Zhao (Keira)
'''
import cPickle
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
            norms = np.sum(transition_matrix, axis=0)
            self._transition_matrix /= norms
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
            norms = np.sum(observation_matrix, axis=0)
            self._observation_matrix /= norms
        else:
            self._observation_matrix = np.eye(num_observ, num_hidden)
        # Build initial distribution, default is uniform distribution
        if initial_dist != None:
            if not (initial_dist.shape[0] == num_hidden):
                raise ValueError("Initial distribution should have length: %d" % num_hidden)
            if not np.all(initial_dist >= 0):
                raise ValueError("Elements in initial_distribution should be non-negative")
            self._initial_dist = initial_dist
            self._initial_dist /= np.sum(initial_dist)
        else:
            self._initial_dist = np.ones(num_hidden, dtype=np.float)
            self._initial_dist /= num_hidden
        # Build accumulative Transition matrix and Observation matrix, which will
        # be useful when generating observation sequences
        self._accumulative_transition_matrix, self._accumulative_observation_matrix = \
        np.add.accumulate(self._transition_matrix, axis=0), \
        np.add.accumulate(self._observation_matrix, axis=0)
        
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
        @note: This method should be overwritten by different subclasses. 
        '''
        pass

    def generate_data(self, dsize, min_seq_len=3, max_seq_len=50):
        '''
        Generate data based on the given HMM.
        @dsize: np.int. Number of observation sequences to be generated
        @min_seq_len: np.int. Minimum length of each observation sequence, inclusive
        @max_seq_len: np.int. Maximum length of each observation sequence, exclusive
        '''
        data = []
        for i in xrange(dsize):
            # Cumulative distribution of the states
            accdist = np.add.accumulate(self._initial_dist)
            rlen = np.random.randint(min_seq_len, max_seq_len)
            sq = np.zeros(rlen, dtype=np.int)
            # Initial state chosen based on the initial distribution
            state = np.where(accdist >= np.random.rand())[0][0]
            for j in xrange(rlen):
                # update the state of HMM by the Transition matrix[state]
                state = np.where(self._accumulative_transition_matrix[:, state] 
                                  >= np.random.rand())[0][0]
                # randomly choose an observation by the Observation matrix[state]
                observ = np.where(self._accumulative_observation_matrix[:, state] 
                                  >= np.random.rand())[0][0]
                sq[j] = observ
            data.append(sq)        
        return data
     
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
    
    ######################################################
    # Static method
    ######################################################
    @staticmethod
    def to_file(filename, hmm):
        with file(filename, "wb") as fout:
            cPickle.dump(hmm, fout)
    
    @staticmethod
    def from_file(filename):
        with file(filename, "rb") as fin:
            model = cPickle.load(fin)
            return model
    

class EMHMM(HMM):
    '''
    This class is used to learning HMM using Baum-Welch algorithm, also
    called Forward-Backward algorithm, which is an implementation of 
    Expectation-Maximization algorithm.
    '''
    def __init__(self, num_hidden, num_observ, 
                 transition_matrix=None, observation_matrix=None, initial_dist=None):
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
                       transition_matrix, observation_matrix, initial_dist)
    
    # Override the fit algorithm provided in HMM, using EM algorithm
    def fit(self, sequences, max_iters=100, repeats=10, seq_length=100,
            para_threshold=None, obj_threshold=None, verbose=False):
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
        if para_threshold == None: para_threshold = self._num_hidden / 500.0
        if obj_threshold == None: obj_threshold = 1e-2
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
        opt_initial_dist = np.random.rand(self._num_hidden)
        opt_log_likelihoods = -np.inf
        for i in xrange(repeats):
            if verbose: 
                pprint("#" * 50)
                pprint("Repeat times: %d" % i)
            #When train is called each time, start with different random points
            self._transition_matrix = np.random.rand(self._num_hidden, self._num_hidden)
            norms = np.sum(self._transition_matrix, axis=0)
            
            self._transition_matrix /= norms
            self._observation_matrix = np.random.rand(self._num_observ, self._num_hidden)
            norms = np.sum(self._observation_matrix, axis=0)
            self._observation_matrix /= norms
            
            self._initial_dist = np.random.rand(self._num_hidden)
            norms = np.sum(self._initial_dist)
            self._initial_dist /= norms
                
            iters = 0
            last_iter_lld = -np.inf
            # Forward-Backward Algorithm to train HMM on seq_lists
            while iters < max_iters:
                iters += 1
                initial_dist = np.zeros(self._num_hidden, dtype=np.float)
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
                    # Updating initial_dist
                    initial_dist += np.sum(sigma_matrix, axis=0)
                    for k in xrange(len(seq) - 1):
                        sigma_matrix = self._transition_matrix * \
                        np.outer(self._observation_matrix[seq[k + 1], :] * beta_matrix[k + 1, :], alpha_matrix[k, :])
                        sigma_matrix /= np.sum(sigma_matrix)
                        transition += sigma_matrix
                        observation[seq[k], :] += np.sum(sigma_matrix, axis=0)
                # Updating transition and observation matrix
                transition /= np.sum(transition, axis=0)[np.newaxis, :]
                observation /= np.sum(observation, axis=0)[np.newaxis, :]
                initial_dist /= np.sum(initial_dist)
                iter_lld = np.sum(np.log([self.predict(seq) for seq in short_sequences]))
                if np.abs(iter_lld-last_iter_lld) < obj_threshold and \
                    np.sum(np.abs(self._transition_matrix - transition)) < para_threshold and \
                    np.sum(np.abs(self._observation_matrix - observation)) < para_threshold:
                    break
                self._transition_matrix = transition
                self._observation_matrix = observation
                self._initial_dist = initial_dist
                last_iter_lld = iter_lld
                if verbose: 
                    pprint("%d iteration: %f" % (iters, iter_lld))
            log_likelihoods = np.sum(np.log([self.predict(seq) for seq in short_sequences]))
            if log_likelihoods > opt_log_likelihoods:
                opt_log_likelihoods = log_likelihoods
                opt_transition_matrix = self._transition_matrix
                opt_observation_matrix = self._observation_matrix
                opt_initial_dist = self._initial_dist
        # Setting parameter which gains the highest log-likelihoods
        if verbose: 
            pprint("Optimal log-likelihood value: %f" % opt_log_likelihoods)
        self._transition_matrix = opt_transition_matrix
        self._observation_matrix = opt_observation_matrix
        self._initial_dist = opt_initial_dist
        

class SLHMM(HMM):
    '''
    This class is used to learning HMM using Spectral Learning algorithm.
    For more detail, please refer to the following paper:
    
    A Spectral Algorithm for Learning Hidden Markov Model, Hsu et al.
    http://arxiv.org/pdf/0811.4413.pdf
    
    Note that the spectral learning algorithm proposed in the paper above
    only supports solving the estimation problem and the learning problem 
    (not learning the transition matrix, observation matrix and initial
    distribution directly, but return a set of transformed observable 
    operators which support computing the marginal joint probability distribution:
    Pr(o_1, o_2,..., o_t))
    '''
    def __init__(self, num_hidden, num_observ, 
                 transition_matrix=None, observation_matrix=None, initial_dist=None):
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
        super(SLHMM, self).__init__(num_hidden, num_observ, 
                       transition_matrix, observation_matrix, initial_dist)
        # First three order moments
        self._P_1 = np.zeros(self._num_observ, dtype=np.float)
        self._P_21 = np.zeros((self._num_observ, self._num_observ), dtype=np.float)
        self._P_3x1 = np.zeros((self._num_observ, self._num_observ, self._num_observ), dtype=np.float)


    # Override the fit algorithm provided in HMM, using Spectral Learning 
    # algorithm
    def fit(self, sequences, rank_hyperparameter=None, verbose=False):
        '''
        Solve the learning problem with HMM.
        @sequences: [np.array]. List of observation sequences, each observation 
                    sequence can have different length
        
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ). 
        '''
        # Set default value of rank-hyperparameter
        if rank_hyperparameter == None:
            rank_hyperparameter = self._num_hidden
        # Training triples
        trilst = np.array([sequence[idx: idx+3] for sequence in sequences
                           for idx in xrange(len(sequence)-2)], dtype=np.int)
        if verbose:
            pprint('Number of separated triples: %d' % trilst.shape[0])
        # Parameter estimation
        # Frequency based estimation
        for sq in trilst:
            self._P_1[sq[0]] += 1
            self._P_21[sq[1], sq[0]] += 1
            self._P_3x1[sq[1], sq[2], sq[0]] += 1
        # Normalization of P_1, P_21, P_3x1
        norms = np.sum(self._P_1)
        self._P_1 /= norms
        # Normalize the joint distribution of P_21        
        norms = np.sum(self._P_21)
        self._P_21 /= norms
        # Normalize the joint distribution of P_3x1
        norms = np.sum(self._P_3x1)
        self._P_3x1 /= norms
        # Singular Value Decomposition
        # Keep all the positive singular values
        (U, _, V) = np.linalg.svd(self._P_21)
        U = U[:, 0:rank_hyperparameter]
        V = V[0:rank_hyperparameter, :]
        # Compute b1, binf and Bx
        # self.factor = (P_21^{T} * U)^{+}, which is used to accelerate the computation
        factor = np.linalg.pinv(np.dot(self._P_21.T, U))
        self._b1 = np.dot(U.T, self._P_1)        
        self._binf = np.dot(factor, self._P_1)
        self._Bx = np.zeros((self._num_observ, rank_hyperparameter, rank_hyperparameter), dtype=np.float)        
        for index in xrange(self._num_observ):
            tmp = np.dot(U.T, self._P_3x1[index])
            self._Bx[index] = np.dot(tmp, factor.T)

    # Overwrite the prediction algorithm using DP provided in base class
    def predict(self, sequence):
        '''
        Solve the estimation problem with HMM.
        @sequence: np.array. Observation sequence
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ). This algorithm uses transformed 
                    observable operator to compute the marginal joint probability
        '''
        prob = self._b1
        for ob in sequence:
            prob = np.dot(self._Bx[ob], prob)
        prob = np.dot(self._binf.T, prob)
        return prob        
