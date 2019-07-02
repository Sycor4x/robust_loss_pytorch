#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019, CrowdStrike, Inc. All rights reserved.
# Author: David J. Elkind
# Creation date: 2019-07-02 (YYYY-MM-DD)


"""
"""

from __future__ import division

import numpy as np

from robust_loss_pytorch.adaptive import AdaptiveLossFunction

if __name__ == "__main__":
  num_dims = 10
  adaptive = AdaptiveLossFunction(num_dims=num_dims,
                                  float_dtype=np.float64,
                                  device="cpu",
                                  alpha_lo=0.001,
                                  alpha_hi=1.999,
                                  alpha_init=None,
                                  scale_lo=1e-5,
                                  scale_init=1.0, )
  foo = adaptive.lossfun(np.random.randn(3,10) - np.random.randn(3,10))
  print(foo.sum())
