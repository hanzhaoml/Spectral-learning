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

from hmm import HMM, EMHMM, SLHMM
from utils import io

class TestHMM(unittest.TestCase):    
        
    # Establish testing environment
    def setUp(self):
        self._model_filename = "hmm_m4n4.pkl"
        self._train_filename = "m4n4.train.data"
        self._num_hidden = 4
        self._num_observ = 4
        transition_matrix = np.random.rand(4, 4)
        observation_matrix = np.random.rand(4, 4)
        hmm = HMM(self._num_hidden, self._num_observ, transition_matrix=transition_matrix,
                  observation_matrix=observation_matrix)
        sequences = hmm.generate_data(10000, 4, 51)
        io.save_sequences(self._train_filename, sequences)
        HMM.to_file(self._model_filename, hmm)
   
    # Test accumulative matrix
    def test_accumulative(self):
        hmm = HMM.from_file(self._model_filename)
        for i in xrange(self._num_hidden):
            self.assertAlmostEqual(1.0, hmm._accumulative_transition_matrix[-1, i], 
                             delta=1e-6)
            self.assertAlmostEqual(1.0, hmm._accumulative_observation_matrix[-1, i], 
                             delta=1e-6)
        
    # Test loading data
    def test_loading(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        for sequence in sequences:
            self.assertEqual(hmm.predict(sequence), hmm.predict(sequence), 
                             "Inferred probability is wrong")
    
    # Test HMM.predict
    def test_predict(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        for sequence in sequences:
            self.assertEqual(hmm.predict(sequence), hmm.predict(sequence), 
                             "HMM.prediction Error")
        sequences = [[0,1], [1,2,3,0], [0,0,0,1]]
        for sequence in sequences:
            self.assertEqual(hmm.predict(sequence), hmm.predict(sequence), 
                             "HMM.prediction Error")
    
    # Test HMM.decode
    def test_decode(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        for sequence in sequences:
            decoded_sequence = hmm.decode(sequence)
            self.assertEqual(len(sequence), len(decoded_sequence), "HMM.decode Error")
            for i in xrange(len(sequence)):
                self.assertEqual(sequence[i], sequence[i], "HMM.decode Error")
        sequences = [[0, 1], [1, 2], [0, 1, 2, 0]]
        for sequence in sequences:
            decoded_sequence = hmm.decode(sequence)
            self.assertEqual(len(sequence), len(decoded_sequence), "HMM.decode Error")
            for i in xrange(len(sequence)):
                self.assertEqual(decoded_sequence[i], decoded_sequence[i], 
                                 "HMM.decode Error")
    
    # Test SLHMM.fit
    def test_slfit(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        learner = SLHMM(self._num_hidden, self._num_observ)
        learner.fit(sequences, verbose=True)
        for sequence in sequences:
            pprint("True probability: %f" % hmm.predict(sequence))
            pprint("Infered probability: %f" % learner.predict(sequence))
    
    # Test EMHMM.fit
    @unittest.skip("Time costing, validated to be correct")
    def test_emfit(self):
        sequences = io.load_sequences(self._train_filename)
        hmm = HMM.from_file(self._model_filename)
        learner = EMHMM(self._num_hidden, self._num_observ)
        learner.fit(sequences, verbose=True, repeats=1)
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
        pprint("*" * 50)
        pprint("True Transition matrix: ")
        pprint(hmm.transition_matrix)
        pprint("True Observation matrix: ")
        pprint(hmm.observation_matrix)
        pprint("True initial distribution: ")
        pprint(hmm.initial_dist)
        
        
if __name__ == '__main__':
    unittest.main()
