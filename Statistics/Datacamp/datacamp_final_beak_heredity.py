import math
import io
import random

import numpy as np
import pandas as pd
from itertools import product, combinations
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import draw_bs_pairs_linreg, draw_bs_reps, ecdf, draw_bs_pairs, pearson_r

bd_parent_scandens = np.array([8.3318, 8.4035, 8.5317, 8.7202, 8.7089, 8.7541, 8.773,
                               8.8107, 8.7919, 8.8069, 8.6523, 8.6146, 8.6938, 8.7127,
                               8.7466, 8.7504, 8.7805, 8.7428, 8.7164, 8.8032, 8.8258,
                               8.856, 8.9012, 8.9125, 8.8635, 8.8258, 8.8522, 8.8974,
                               8.9427, 8.9879, 8.9615, 8.9238, 8.9351, 9.0143, 9.0558,
                               9.0596, 8.9917, 8.905, 8.9314, 8.9465, 8.9879, 8.9804,
                               9.0219, 9.052, 9.0407, 9.0407, 8.9955, 8.9992, 8.9992,
                               9.0747, 9.0747, 9.5385, 9.4781, 9.4517, 9.3537, 9.2707,
                               9.1199, 9.1689, 9.1425, 9.135, 9.1011, 9.1727, 9.2217,
                               9.2255, 9.2821, 9.3235, 9.3198, 9.3198, 9.3198, 9.3273,
                               9.3725, 9.3989, 9.4253, 9.4593, 9.4442, 9.4291, 9.2632,
                               9.2293, 9.1878, 9.1425, 9.1275, 9.1802, 9.1765, 9.2481,
                               9.2481, 9.1991, 9.1689, 9.1765, 9.2406, 9.3198, 9.3235,
                               9.1991, 9.2971, 9.2443, 9.316, 9.2934, 9.3914, 9.3989,
                               9.5121, 9.6176, 9.5535, 9.4668, 9.3725, 9.3348, 9.3763,
                               9.3839, 9.4216, 9.4065, 9.3348, 9.4442, 9.4367, 9.5083,
                               9.448, 9.4781, 9.595, 9.6101, 9.5686, 9.6365, 9.7119,
                               9.8213, 9.825, 9.7609, 9.6516, 9.5988, 9.546, 9.6516,
                               9.7572, 9.8854, 10.0023, 9.3914])

bd_offspring_scandens = np.array([8.419, 9.2468, 8.1532, 8.0089, 8.2215, 8.3734, 8.5025,
                                  8.6392, 8.7684, 8.8139, 8.7911, 8.9051, 8.9203, 8.8747,
                                  8.943, 9.0038, 8.981, 9.0949, 9.2696, 9.1633, 9.1785,
                                  9.1937, 9.2772, 9.0722, 8.9658, 8.9658, 8.5025, 8.4949,
                                  8.4949, 8.5633, 8.6013, 8.6468, 8.1532, 8.3734, 8.662,
                                  8.6924, 8.7456, 8.8367, 8.8595, 8.9658, 8.9582, 8.8671,
                                  8.8671, 8.943, 9.0646, 9.1405, 9.2089, 9.2848, 9.3759,
                                  9.4899, 9.4519, 8.1228, 8.2595, 8.3127, 8.4949, 8.6013,
                                  8.4646, 8.5329, 8.7532, 8.8823, 9.0342, 8.6392, 8.6772,
                                  8.6316, 8.7532, 8.8291, 8.8975, 8.9734, 9.0494, 9.1253,
                                  9.1253, 9.1253, 9.1785, 9.2848, 9.4595, 9.3608, 9.2089,
                                  9.2544, 9.3684, 9.3684, 9.2316, 9.1709, 9.2316, 9.0342,
                                  8.8899, 8.8291, 8.981, 8.8975, 10.4089, 10.1886, 9.7633,
                                  9.7329, 9.6114, 9.5051, 9.5127, 9.3684, 9.6266, 9.5354,
                                  10.0215, 10.0215, 9.6266, 9.6038, 9.4063, 9.2316, 9.338,
                                  9.262, 9.262, 9.4063, 9.4367, 9.0342, 8.943, 8.9203,
                                  8.7835, 8.7835, 9.057, 8.9354, 8.8975, 8.8139, 8.8671,
                                  9.0873, 9.2848, 9.2392, 9.2924, 9.4063, 9.3152, 9.4899,
                                  9.5962, 9.6873, 9.5203, 9.6646])

bd_parent_fortis = np.array([10.1, 9.55, 9.4, 10.25, 10.125, 9.7, 9.05, 7.4,
                             9., 8.65, 9.625, 9.9, 9.55, 9.05, 8.35, 10.1,
                             10.1, 9.9, 10.225, 10., 10.55, 10.45, 9.2, 10.2,
                             8.95, 10.05, 10.2, 9.5, 9.925, 9.95, 10.05, 8.75,
                             9.2, 10.15, 9.8, 10.7, 10.5, 9.55, 10.55, 10.475,
                             8.65, 10.7, 9.1, 9.4, 10.3, 9.65, 9.5, 9.7,
                             10.525, 9.95, 10.1, 9.75, 10.05, 9.9, 10., 9.1,
                             9.45, 9.25, 9.5, 10., 10.525, 9.9, 10.4, 8.95,
                             9.4, 10.95, 10.75, 10.1, 8.05, 9.1, 9.55, 9.05,
                             10.2, 10., 10.55, 10.75, 8.175, 9.7, 8.8, 10.75,
                             9.3, 9.7, 9.6, 9.75, 9.6, 10.45, 11., 10.85,
                             10.15, 10.35, 10.4, 9.95, 9.1, 10.1, 9.85, 9.625,
                             9.475, 9., 9.25, 9.1, 9.25, 9.2, 9.95, 8.65,
                             9.8, 9.4, 9., 8.55, 8.75, 9.65, 8.95, 9.15,
                             9.85, 10.225, 9.825, 10., 9.425, 10.4, 9.875, 8.95,
                             8.9, 9.35, 10.425, 10., 10.175, 9.875, 9.875, 9.15,
                             9.45, 9.025, 9.7, 9.7, 10.05, 10.3, 9.6, 10.,
                             9.8, 10.05, 8.75, 10.55, 9.7, 10., 9.85, 9.8,
                             9.175, 9.65, 9.55, 9.9, 11.55, 11.3, 10.4, 10.8,
                             9.8, 10.45, 10., 10.75, 9.35, 10.75, 9.175, 9.65,
                             8.8, 10.55, 10.675, 9.95, 9.55, 8.825, 9.7, 9.85,
                             9.8, 9.55, 9.275, 10.325, 9.15, 9.35, 9.15, 9.65,
                             10.575, 9.975, 9.55, 9.2, 9.925, 9.2, 9.3, 8.775,
                             9.325, 9.175, 9.325, 8.975, 9.7, 9.5, 10.225, 10.025,
                             8.2, 8.2, 9.55, 9.05, 9.6, 9.6, 10.15, 9.875,
                             10.485, 11.485, 10.985, 9.7, 9.65, 9.35, 10.05, 10.1,
                             9.9, 8.95, 9.3, 9.95, 9.45, 9.5, 8.45, 8.8,
                             8.525, 9.375, 10.2, 7.625, 8.375, 9.25, 9.4, 10.55,
                             8.9, 8.8, 9., 8.575, 8.575, 9.6, 9.375, 9.6,
                             9.95, 9.6, 10.2, 9.85, 9.625, 9.025, 10.375, 10.25,
                             9.3, 9.5, 9.55, 8.55, 9.05, 9.9, 9.8, 9.75,
                             10.25, 9.1, 9.65, 10.3, 8.9, 9.95, 9.5, 9.775,
                             9.425, 7.75, 7.55, 9.1, 9.6, 9.575, 8.95, 9.65,
                             9.65, 9.65, 9.525, 9.85, 9.05, 9.3, 8.9, 9.45,
                             10., 9.85, 9.25, 10.1, 9.125, 9.65, 9.1, 8.05,
                             7.4, 8.85, 9.075, 9., 9.7, 8.7, 9.45, 9.7,
                             8.35, 8.85, 9.7, 9.45, 10.3, 10., 10.45, 9.45,
                             8.5, 8.3, 10., 9.225, 9.75, 9.15, 9.55, 9.,
                             9.275, 9.35, 8.95, 9.875, 8.45, 8.6, 9.7, 8.55,
                             9.05, 9.6, 8.65, 9.2, 8.95, 9.6, 9.15, 9.4,
                             8.95, 9.95, 10.55, 9.7, 8.85, 8.8, 10., 9.05,
                             8.2, 8.1, 7.25, 8.3, 9.15, 8.6, 9.5, 8.05,
                             9.425, 9.3, 9.8, 9.3, 9.85, 9.5, 8.65, 9.825,
                             9., 10.45, 9.1, 9.55, 9.05, 10., 9.35, 8.375,
                             8.3, 8.8, 10.1, 9.5, 9.75, 10.1, 9.575, 9.425,
                             9.65, 8.725, 9.025, 8.5, 8.95, 9.3, 8.85, 8.95,
                             9.8, 9.5, 8.65, 9.1, 9.4, 8.475, 9.35, 7.95,
                             9.35, 8.575, 9.05, 8.175, 9.85, 7.85, 9.85, 10.1,
                             9.35, 8.85, 8.75, 9.625, 9.25, 9.55, 10.325, 8.55,
                             9.675, 9.15, 9., 9.65, 8.6, 8.8, 9., 9.95,
                             8.4, 9.35, 10.3, 9.05, 9.975, 9.975, 8.65, 8.725,
                             8.2, 7.85, 8.775, 8.5, 9.4])

bd_offspring_fortis = np.array([10.7 ,  9.78,  9.48,  9.6 , 10.27,  9.5 ,  9.  ,  7.46,  7.65,
                                8.63,  9.81,  9.4 ,  9.48,  8.75,  7.6 , 10.  , 10.09,  9.74,
                                9.64,  8.49, 10.15, 10.28,  9.2 , 10.01,  9.03,  9.94, 10.5 ,
                                9.7 , 10.02, 10.04,  9.43,  8.1 ,  9.5 ,  9.9 ,  9.48, 10.18,
                                10.16,  9.08, 10.39,  9.9 ,  8.4 , 10.6 ,  8.75,  9.46,  9.6 ,
                                9.6 ,  9.95, 10.05, 10.16, 10.1 ,  9.83,  9.46,  9.7 ,  9.82,
                                10.34,  8.02,  9.65,  9.87,  9.  , 11.14,  9.25,  8.14, 10.23,
                                8.7 ,  9.8 , 10.54, 11.19,  9.85,  8.1 ,  9.3 ,  9.34,  9.19,
                                9.52,  9.36,  8.8 ,  8.6 ,  8.  ,  8.5 ,  8.3 , 10.38,  8.54,
                                8.94, 10.  ,  9.76,  9.45,  9.89, 10.9 ,  9.91,  9.39,  9.86,
                                9.74,  9.9 ,  9.09,  9.69, 10.24,  8.9 ,  9.67,  8.93,  9.3 ,
                                8.67,  9.15,  9.23,  9.59,  9.03,  9.58,  8.97,  8.57,  8.47,
                                8.71,  9.21,  9.13,  8.5 ,  9.58,  9.21,  9.6 ,  9.32,  8.7 ,
                                10.46,  9.29,  9.24,  9.45,  9.35, 10.19,  9.91,  9.18,  9.89,
                                9.6 , 10.3 ,  9.45,  8.79,  9.2 ,  8.8 ,  9.69, 10.61,  9.6 ,
                                9.9 ,  9.26, 10.2 ,  8.79,  9.28,  8.83,  9.76, 10.2 ,  9.43,
                                9.4 ,  9.9 ,  9.5 ,  8.95,  9.98,  9.72,  9.86, 11.1 ,  9.14,
                                10.49,  9.75, 10.35,  9.73,  9.83,  8.69,  9.58,  8.42,  9.25,
                                10.12,  9.31,  9.99,  8.59,  8.74,  8.79,  9.6 ,  9.52,  8.93,
                                10.23,  9.35,  9.35,  9.09,  9.04,  9.75, 10.5 ,  9.09,  9.05,
                                9.54,  9.3 ,  9.06,  8.7 ,  9.32,  8.4 ,  8.67,  8.6 ,  9.53,
                                9.77,  9.65,  9.43,  8.35,  8.26,  9.5 ,  8.6 ,  9.57,  9.14,
                                10.79,  8.91,  9.93, 10.7 ,  9.3 ,  9.93,  9.51,  9.44, 10.05,
                                10.13,  9.24,  8.21,  8.9 ,  9.34,  8.77,  9.4 ,  8.82,  8.83,
                                8.6 ,  9.5 , 10.2 ,  8.09,  9.07,  9.29,  9.1 , 10.19,  9.25,
                                8.98,  9.02,  8.6 ,  8.25,  8.7 ,  9.9 ,  9.65,  9.45,  9.38,
                                10.4 ,  9.96,  9.46,  8.26, 10.05,  8.92,  9.5 ,  9.43,  8.97,
                                8.44,  8.92, 10.3 ,  8.4 ,  9.37,  9.91, 10.  ,  9.21,  9.95,
                                8.84,  9.82,  9.5 , 10.29,  8.4 ,  8.31,  9.29,  8.86,  9.4 ,
                                9.62,  8.62,  8.3 ,  9.8 ,  8.48,  9.61,  9.5 ,  9.37,  8.74,
                                9.31,  9.5 ,  9.49,  9.74,  9.2 ,  9.24,  9.7 ,  9.64,  9.2 ,
                                7.5 ,  7.5 ,  8.7 ,  8.31,  9.  ,  9.74,  9.31, 10.5 ,  9.3 ,
                                8.12,  9.34,  9.72,  9.  ,  9.65,  9.9 , 10.  , 10.1 ,  8.  ,
                                9.07,  9.75,  9.33,  8.11,  9.36,  9.74,  9.9 ,  9.23,  9.7 ,
                                8.2 ,  9.35,  9.49,  9.34,  8.87,  9.03,  9.07,  9.43,  8.2 ,
                                9.19,  9.  ,  9.2 ,  9.06,  9.81,  8.89,  9.4 , 10.45,  9.64,
                                9.03,  8.71,  9.91,  8.33,  8.2 ,  7.83,  7.14,  8.91,  9.18,
                                8.8 ,  9.9 ,  7.73,  9.25,  8.7 ,  9.5 ,  9.3 ,  9.05, 10.18,
                                8.85,  9.24,  9.15,  9.98,  8.77,  9.8 ,  8.65, 10.  ,  8.81,
                                8.01,  7.9 ,  9.41, 10.18,  9.55,  9.08,  8.4 ,  9.75,  8.9 ,
                                9.07,  9.35,  8.9 ,  8.19,  8.65,  9.19,  8.9 ,  9.28, 10.58,
                                9.  ,  9.4 ,  8.91,  9.93, 10.  ,  9.37,  7.4 ,  9.  ,  8.8 ,
                                9.18,  8.3 , 10.08,  7.9 ,  9.96, 10.4 ,  9.65,  8.8 ,  8.65,
                                9.7 ,  9.23,  9.43,  9.93,  8.47,  9.55,  9.28,  8.85,  8.9 ,
                                8.75,  8.63,  9.  ,  9.43,  8.28,  9.23, 10.4 ,  9.  ,  9.8 ,
                                9.77,  8.97,  8.37,  7.7 ,  7.9 ,  9.5 ,  8.2 ,  8.8 ])


plt.style.use('ggplot')
"""

  Tasks 

"""


def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1] / covariance_matrix[0,0]


def get_covariances_of_permutated_parent():
    # my null hypothesis here is that if a scramble parent reads and compare them to offspring
    # I'd get the same level of correlation between the observed covariance
    # meaning that there was no real correlation between parent and offspring
    # but the low p-value here indicates that that is NOT true,
    #  and so we reject the hypothesis that there is no correlation
    # and support the alternate that there is in fact some correlation between parent and offspring

    heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(10000)

    # Draw replicates
    for i in range(10000):
        # Permute parent beak depths
        bd_parent_permuted = np.random.permutation(bd_parent_scandens)
        perm_replicates[i] = heritability(bd_parent_permuted, bd_offspring_scandens)

    # Compute p-value: p
    p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)

    # Print the p-value
    print('p-val =', p)


def analyse_covariances():
    print("\nAnalyse by covariance")

    heritability_scandens = heritability(bd_parent_scandens, bd_offspring_scandens)
    heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)
    
    # Acquire 1000 bootstrap replicates of heritability
    replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)
    replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)

    # Compute 95% confidence intervals
    conf_int_scandens = np.percentile(replicates_scandens, [2.5, 97.5])
    conf_int_fortis = np.percentile(replicates_fortis, [2.5, 97.5])
    
    # Print results
    print('G. scandens:', heritability_scandens, conf_int_scandens)
    print('G. fortis:', heritability_fortis, conf_int_fortis)


def plot_parent_offspring_trend():
    # Make scatter plots
    _ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
                 marker='.', linestyle='none', color='r', alpha=0.5)

    _ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
                 marker='x', linestyle='none', color='b', alpha=0.5)

    # Label axes
    _ = plt.xlabel('parental beak depth (mm)')
    _ = plt.ylabel('offspring beak depth (mm)')

    # Add legend
    _ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')

    # Show plot
    plt.show()


def apply_pearsons():
    print("\nAnalyse by corrcoef")

    # Compute the Pearson correlation coefficients
    r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
    r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

    # Acquire 1000 bootstrap replicates of Pearson r
    bs_replicates_scandens = draw_bs_pairs(x=bd_parent_scandens, y=bd_offspring_scandens, func=pearson_r, size=1000)
    bs_replicates_fortis = draw_bs_pairs(x=bd_parent_fortis, y=bd_offspring_fortis, func=pearson_r, size=1000)

    # Compute 95% confidence intervals
    conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
    conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])

    # Print results
    print('G. scandens:', r_scandens, conf_int_scandens)
    print('G. fortis:', r_fortis, conf_int_fortis)


"""

Run
 
"""
if __name__ == '__main__':
    plot_parent_offspring_trend()
    apply_pearsons()
    analyse_covariances()
    get_covariances_of_permutated_parent()
