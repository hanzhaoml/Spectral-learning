#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.

import numpy as np
import csv
import time
import sys
from hmm import HMM
from learner import SpectralLearner
from EM import BaumWelch


    
if __name__ == '__main__':
    usage = './SPClassifier modelpath trainset testset'
    if len(sys.argv) < 4:
        print usage
        exit()
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3]) 
    