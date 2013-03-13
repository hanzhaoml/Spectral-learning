#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) 2013 KeiraZhao <zhaohan2009011312@gmail.com>
#
# Distributed under terms of the Tsinghua University license.

import sys
import random

if __name__ == '__main__':
    
    assert len(sys.argv) >= 3
    totlen = 1000000
    size = int(float(sys.argv[2]) * totlen)
    
    with file(sys.argv[1], 'r') as f:
        array = range(totlen)
        random.shuffle(array)
        array = set(array[:size])
        filename = sys.argv[2] + '_' + sys.argv[1]
        fout = file(filename, 'w')
        
        for idx, line in enumerate(f):
            if idx in array:
                fout.write(line)
        fout.close()
            
            
        