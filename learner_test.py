#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.
import unittest

import numpy as np

from learner import BoostedLearner

class Test(unittest.TestCase):

    def setUp(self):
        self.seq = [[1, 2, 3], [1, 2, 3],
                    [1, 3, 2], [2, 1, 3],
                    [3, 2, 1], [4, 9, 2]]
        self.learner = BoostedLearner()
        
    @unittest.skip('learner.BoostedLearner.boosting() has not been finished...')
    def testBuilding(self):
        (triples, weights) = self.learner.boosting(self.seq)
        for idx in xrange(len(triples)):
            print triples[idx], weights[idx]
        print '=' * 50
        
    def testRandom(self):
        print 'Starting sampling...'
        avg = np.random.rand(6)
        avg /= np.sum(avg)
        print avg
        weights = np.random.multinomial(200000, avg.tolist(), size=1).tolist()
        print weights
        self.assertEqual(200000, np.sum(weights), 'These two sums are not equal')
        print 'End sampling, start averaging...'
        weights = np.sum(weights, axis=0)
        weights.dtype = np.float
        weights /= np.sum(weights)
        print weights
        

if __name__ == "__main__":
    unittest.main()




