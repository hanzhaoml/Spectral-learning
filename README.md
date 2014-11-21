Spectral-learning
=================
Expectation Maximization and Spectral Learning for Hidden Markov Model

Expectation Maximization
------------------------
BaumWelch (Forward-backward) learning algorithm for fitting the data.

Spectral Learning
------------------------
LearnHMM utilizes only upto first three order moments and take advantage
of the moment matching idea. Please refer to Daniel Hsu's paper (A
spectral learning algorithm for Hidden Markov Models) for more details.
Note that there is a well-known issue, called the negative probability
problem associated spectral learning. In this implement we leave the 
problem as is.
