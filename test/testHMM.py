#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.
import sys, os
import unittest
import numpy as np

from pprint import pprint

sys.path.append("..")
sys.path.append(os.path.join("..", "hmm"))
sys.path.append(os.path.join("..", "utils"))

from hmm import HMM, EMHMM
from utils import io

class TestHMM(unittest.TestCase):    
        
    # Establish testing environment
    def setUp(self):
        self._model_filename = "hmm_m4n4.pkl"
        self._train_filename = "m4n4.train.data"
        self._num_hidden = 4
        self._num_observ = 4
#         transition_matrix = np.random.rand(4, 4)
#         observation_matrix = np.random.rand(4, 4)
#         hmm = HMM(num_hidden, num_observ, transition_matrix=transition_matrix,
#                   observation_matrix=observation_matrix)
        hmm = HMM(self._num_hidden, self._num_observ)
        sequences = hmm.generate_data(100, 4, 51)
        io.save_sequences(self._train_filename, sequences)
        HMM.to_file(self._model_filename, hmm)
   
    # Test loading data
    def test_loading(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        for sequence in sequences:
            self.assertEqual(0.25, hmm.predict(sequence), 
                             "Inferred probability is wrong")
    
    # Test HMM.predict
    def test_predict(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        for sequence in sequences:
            self.assertEqual(0.25, hmm.predict(sequence), "HMM.prediction Error")
        sequences = [[0,1], [1,2,3,0], [0,0,0,1]]
        for sequence in sequences:
            self.assertEqual(0, hmm.predict(sequence), "HMM.prediction Error")
    
    # Test HMM.decode
    def test_decode(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        for sequence in sequences:
            decoded_sequence = hmm.decode(sequence)
            self.assertEqual(len(sequence), len(decoded_sequence), "HMM.decode Error")
            for i in xrange(len(sequence)):
                self.assertEqual(sequence[i], decoded_sequence[i], "HMM.decode Error")
        sequences = [[0, 1], [1, 2], [0, 1, 2, 0]]
        for sequence in sequences:
            decoded_sequence = hmm.decode(sequence)
            self.assertEqual(len(sequence), len(decoded_sequence), "HMM.decode Error")
            for i in xrange(len(sequence)):
                self.assertEqual(decoded_sequence[i], 0, "HMM.decode Error")
    
    # Test HMM.fit
    def test_fit(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        learner = EMHMM(self._num_hidden, self._num_observ)
        learner.fit(sequences, verbose=True)
        for sequence in sequences:
            pprint("True probability: %f" % hmm.predict(sequence))
            pprint("Infered probability: %f" % learner.predict(sequence))
        pprint("Learned parameter using EM algorithm:")
        pprint("Transition matrix: ")
        pprint(learner.transition_matrix)
        pprint("Observation matrix: ")
        pprint(learner.observation_matrix)
        pprint("Initial distribution: ")
        pprint(learner.initial_dist)
        
# def testBaumWelch(training_filename, test_filename, model_filename):
#     '''
#     @training_filename: string. Path to training set
#     @test_filename: string. Path to test set
#     @model_filename: string. Path to the HMM which generates the training
#                      and test set
#     '''
#     hmm = HMM.from_file(model_filename)
#     training_data = np.loadtxt(training_filename, dtype=np.int, delimiter=",")
#     pprint("Start training HMM with EM...")
#     for i in xrange(20):
#         start_time = time.clock()
#         learner = BaumWelch(hmm.m, hmm.n)
#         learner.train(training_data)
#         end_time = time.clock()
#         pprint("=" * 50)
#         pprint("EM %d th running" % (i + 1))
#         pprint("Total time used to train HMM with EM: %f" % (end_time - start_time))
#         pprint("Stationary distribution: ")
#         pprint(learner.stationary_dist)
#         pprint("Transition matrix: ")
#         pprint(learner.transition_matrix)
#         pprint("Observation matrix: ")
#         pprint(learner.observation_matrix)
# 
# 
# def EM_consistency(training_filename, test_filename, model_filename):
#     hmm = HMM.from_file(model_filename)
#     training_data = np.loadtxt(training_filename, dtype=np.int, delimiter=",")
#     test_data = []
#     learner = BaumWelch(hmm.m, hmm.n)
#     with file(test_filename, "rb") as fin:
#         reader = csv.reader(fin)
#         for line in reader:
#             test_data.append(np.asarray(map(int, line)))
# #     Setting parameters to start with EM-consistency toy experiment
#     num_chunks = 1000
#     num_segments = 20
#     # num_segments = training_data.shape[0] / num_chunks
#     num_restarts = 20
#     mean_log_prob = np.zeros(num_segments, dtype=np.float)
#     std_log_prob = np.zeros(num_segments, dtype=np.float)
# #     Computing the log-likelihood function value using true model parameters
#     true_log_prob = np.sum(np.log([hmm.probability(x) for x in test_data]))
# #     Train HMM with EM algorithm and then compute the mean log-probability and its
# #     corresponding standard deviation
#     for i in xrange(1, num_segments + 1):
#         cur_training_data = training_data[: i * num_chunks]
#         log_probs = np.zeros(num_restarts, dtype=np.float)
#         for j in xrange(num_restarts):
#             learner.train(cur_training_data)
#             log_probs[j] = np.sum(np.log([learner.predict(x) for x in test_data]))
#         mean_log_prob[i - 1] = np.mean(log_probs)
#         std_log_prob[i - 1] = np.std(log_probs)
#         pprint("-" * 50)
#         pprint("Current Training Size: %d" % (i * num_chunks))
#         pprint("True log-likelihood: %f" % true_log_prob)
#         pprint("EM mean log-likelihood: %f" % mean_log_prob[i - 1])
#         pprint("EM standard deviation of log-likelihood: %f" % std_log_prob[i - 1])
# #     Repeat true_log_prob to form a vector of the same length as mean_log_prob and std_log_prob
#     true_log_prob = np.repeat(true_log_prob, num_segments)
#     return np.array([true_log_prob, mean_log_prob, std_log_prob]).T
#    
# def EM_consistency_same(training_filename, test_filename, model_filename):
#     hmm = HMM.from_file(model_filename)
#     training_data = np.loadtxt(training_filename, dtype=np.int, delimiter=",")
#     learner = BaumWelch(hmm.m, hmm.n)
#     # Training parameters
#     num_chunks = 10
#     num_segments = 20
#     num_restarts = 20
#     mean_log_prob = np.zeros(num_segments, dtype=np.float)
#     std_log_prob = np.zeros(num_segments, dtype=np.float)
#     true_log_prob = np.zeros(num_segments, dtype=np.float)
#     # Computing the log-likelihood function value using true model parameters
#     # Train HMM with EM algorithm and then compute the mean log-probability and its
#     # Corresponding standard deviation
#     for i in xrange(1, num_segments + 1):
#         cur_training_data = training_data[: i * num_chunks]
#         # Partition a long training sequence into set of short sequences
#         seq_length = 100
#         num_partitions = len(cur_training_data) / seq_length
#         test_data = [cur_training_data[k * seq_length: (k + 1) * seq_length] for k in xrange(num_partitions)]
#         if len(cur_training_data) > num_partitions * seq_length:
#             test_data.append(cur_training_data[num_partitions * seq_length: ])
#         # Compute the true log-likelihood
#         true_log_prob[i - 1] = np.sum(np.log([hmm.probability(x) for x in test_data]))
#         log_probs = np.zeros(num_restarts, dtype=np.float)
#         for j in xrange(num_restarts):
#             learner.train(cur_training_data)
#             log_probs[j] = np.sum(np.log([learner.predict(x) for x in test_data]))
#         mean_log_prob[i - 1] = np.mean(log_probs)
#         std_log_prob[i - 1] = np.std(log_probs)
#         pprint("-" * 50)
#         pprint("Value of i: %d" % i)
#         pprint("Current training length: %d" % cur_training_data.shape[0])
#         pprint("Current Training Size: %d" % (i * num_chunks))
#         pprint("True log-likelihood: %f" % true_log_prob[i - 1])
#         pprint("EM mean log-likelihood: %f" % mean_log_prob[i - 1])
#         pprint("EM standard deviation of log-likelihood: %f" % std_log_prob[i - 1])
#     return np.array([true_log_prob, mean_log_prob, std_log_prob]).T
#    
# def easy_to_distinguish(training_filename, test_filename, model_filename):
#     num_hidden = 2
#     num_observ = 2
#     num_training_insts = 2000
#     hmm = HMM(num_hidden, num_observ)
#     hmm.T = np.array([[0.91, 0.09], [0.15, 0.85]], dtype=np.float)
#     hmm.O = np.array([[0.73, 0.27], [0.19, 0.81]], dtype=np.float)
#     HMM.to_file(model_filename, hmm)
#     training_data = hmm.generate_train_data(num_training_insts)
#     np.savetxt(training_filename, [training_data], delimiter=",", fmt="%d")
#     pprint("FInished generating sample data...")
#     pprint("Initial distribution of HMM:")
#     pprint(hmm.stationary_dist)
#     pprint("Transition matrix:")
#     pprint(hmm.transition_matrix)
#     pprint("Observation matrix: ")
#     pprint(hmm.observation_matrix)
# 
# def main(training_filename, test_filename, model_filename):
#     num_hidden = 2
#     num_observ = 2
#     num_training_insts = 100000
#     num_test_insts = 10000
#     hmm = HMM(num_hidden, num_observ)
#     HMM.to_file(model_filename, hmm)
#     training_data = hmm.generate_train_data(num_training_insts)
#     test_data = hmm.generate_test_data(num_test_insts, min_seq_len=4, max_seq_len=10)
#     np.savetxt(training_filename, [training_data], delimiter=",", fmt="%d")
#     with file(test_filename, "wb") as fout:
#         writer = csv.writer(fout)
#         for test_seq in test_data:
#             writer.writerow(test_seq)
#     pprint("Finished generating sample data...")
#     pprint("Initial distribution of HMM: ")
#     pprint(hmm.stationary_dist)
#     pprint("Transition matrix: ")
#     pprint(hmm.transition_matrix)
#     pprint("Observation matrix: ")
#     pprint(hmm.observation_matrix)
    
    
if __name__ == '__main__':
    unittest.main()
#     usage = './test training_filename, test_filename, model_filename log_filename'
#     if len(sys.argv) < 5:
#         print usage
#         exit()
#     training_filename = sys.argv[1]
#     test_filename = sys.argv[2]
#     model_filename = sys.argv[3]
#     log_filename = sys.argv[4]
#     testBaumWelch(training_filename, test_filename, model_filename) 
#     main(training_filename, test_filename, model_filename)
    # statistics = EM_consistency(training_filename, test_filename, model_filename)
#     statistics = EM_consistency_same(training_filename, test_filename, model_filename)
#     np.savetxt(log_filename, statistics, delimiter=",", fmt="%e")
    # easy_to_distinguish(training_filename, test_filename, model_filename)
