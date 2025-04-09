'''
Samples from the Maxwell Boltzmann Distribution to give a particle an initial velocity. Assumes distribution in one dimension at a time.

This is a standard normal distribution with width sqrt(1/(m\beta))
'''
import numpy as np
import constants

def sample_maxwell_boltzmann_distribution():
    return np.random.normal(0, np.sqrt(constants.k_b*constants.T/constants.m))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    count = 10000000
    samples = np.zeros(count)
    for i in range(count):
        samples[i] = sample_maxwell_boltzmann_distribution()
    print(sum(samples)/count)
    plt.hist(samples,bins=100)
    plt.show()