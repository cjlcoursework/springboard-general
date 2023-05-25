import math
import io
import random

import numpy as np
import pandas as pd
from itertools import product, combinations
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from springboard_dc.exercises_datacamp.utilities import ecdf, draw_bs_reps

df = pd.DataFrame(
    {'beak_depth': {0: 8.4,
                    1: 8.8,
                    2: 8.4,
                    3: 8.0,
                    4: 7.9,
                    5: 8.9,
                    6: 8.6,
                    7: 8.5,
                    8: 8.9,
                    9: 9.1,
                    10: 8.6,
                    11: 9.8,
                    12: 8.2,
                    13: 9.0,
                    14: 9.7,
                    15: 8.6,
                    16: 8.2,
                    17: 9.0,
                    18: 8.4,
                    19: 8.6,
                    20: 8.9,
                    21: 9.1,
                    22: 8.3,
                    23: 8.7,
                    24: 9.6,
                    25: 8.5,
                    26: 9.1,
                    27: 9.0,
                    28: 9.2,
                    29: 9.9,
                    30: 8.6,
                    31: 9.2,
                    32: 8.4,
                    33: 8.9,
                    34: 8.5,
                    35: 10.4,
                    36: 9.6,
                    37: 9.1,
                    38: 9.3,
                    39: 9.3,
                    40: 8.8,
                    41: 8.3,
                    42: 8.8,
                    43: 9.1,
                    44: 10.1,
                    45: 8.9,
                    46: 9.2,
                    47: 8.5,
                    48: 10.2,
                    49: 10.1,
                    50: 9.2,
                    51: 9.7,
                    52: 9.1,
                    53: 8.5,
                    54: 8.2,
                    55: 9.0,
                    56: 9.3,
                    57: 8.0,
                    58: 9.1,
                    59: 8.1,
                    60: 8.3,
                    61: 8.7,
                    62: 8.8,
                    63: 8.6,
                    64: 8.7,
                    65: 8.0,
                    66: 8.8,
                    67: 9.0,
                    68: 9.1,
                    69: 9.74,
                    70: 9.1,
                    71: 9.8,
                    72: 10.4,
                    73: 8.3,
                    74: 9.44,
                    75: 9.04,
                    76: 9.0,
                    77: 9.05,
                    78: 9.65,
                    79: 9.45,
                    80: 8.65,
                    81: 9.45,
                    82: 9.45,
                    83: 9.05,
                    84: 8.75,
                    85: 9.45,
                    86: 8.35,
                    87: 9.4,
                    88: 8.9,
                    89: 9.5,
                    90: 11.0,
                    91: 8.7,
                    92: 8.4,
                    93: 9.1,
                    94: 8.7,
                    95: 10.2,
                    96: 9.6,
                    97: 8.85,
                    98: 8.8,
                    99: 9.5,
                    100: 9.2,
                    101: 9.0,
                    102: 9.8,
                    103: 9.3,
                    104: 9.0,
                    105: 10.2,
                    106: 7.7,
                    107: 9.0,
                    108: 9.5,
                    109: 9.4,
                    110: 8.0,
                    111: 8.9,
                    112: 9.4,
                    113: 9.5,
                    114: 8.0,
                    115: 10.0,
                    116: 8.95,
                    117: 8.2,
                    118: 8.8,
                    119: 9.2,
                    120: 9.4,
                    121: 9.5,
                    122: 8.1,
                    123: 9.5,
                    124: 8.4,
                    125: 9.3,
                    126: 9.3,
                    127: 9.6,
                    128: 9.2,
                    129: 10.0,
                    130: 8.9,
                    131: 10.5,
                    132: 8.9,
                    133: 8.6,
                    134: 8.8,
                    135: 9.15,
                    136: 9.5,
                    137: 9.1,
                    138: 10.2,
                    139: 8.4,
                    140: 10.0,
                    141: 10.2,
                    142: 9.3,
                    143: 10.8,
                    144: 8.3,
                    145: 7.8,
                    146: 9.8,
                    147: 7.9,
                    148: 8.9,
                    149: 7.7,
                    150: 8.9,
                    151: 9.4,
                    152: 9.4,
                    153: 8.5,
                    154: 8.5,
                    155: 9.6,
                    156: 10.2,
                    157: 8.8,
                    158: 9.5,
                    159: 9.3,
                    160: 9.0,
                    161: 9.2,
                    162: 8.7,
                    163: 9.0,
                    164: 9.1,
                    165: 8.7,
                    166: 9.4,
                    167: 9.8,
                    168: 8.6,
                    169: 10.6,
                    170: 9.0,
                    171: 9.5,
                    172: 8.1,
                    173: 9.3,
                    174: 9.6,
                    175: 8.5,
                    176: 8.2,
                    177: 8.0,
                    178: 9.5,
                    179: 9.7,
                    180: 9.9,
                    181: 9.1,
                    182: 9.5,
                    183: 9.8,
                    184: 8.4,
                    185: 8.3,
                    186: 9.6,
                    187: 9.4,
                    188: 10.0,
                    189: 8.9,
                    190: 9.1,
                    191: 9.8,
                    192: 9.3,
                    193: 9.9,
                    194: 8.9,
                    195: 8.5,
                    196: 10.6,
                    197: 9.3,
                    198: 8.9,
                    199: 8.9,
                    200: 9.7,
                    201: 9.8,
                    202: 10.5,
                    203: 8.4,
                    204: 10.0,
                    205: 9.0,
                    206: 8.7,
                    207: 8.8,
                    208: 8.4,
                    209: 9.3,
                    210: 9.8,
                    211: 8.9,
                    212: 9.8,
                    213: 9.1},
     'year': {0: 1975,
              1: 1975,
              2: 1975,
              3: 1975,
              4: 1975,
              5: 1975,
              6: 1975,
              7: 1975,
              8: 1975,
              9: 1975,
              10: 1975,
              11: 1975,
              12: 1975,
              13: 1975,
              14: 1975,
              15: 1975,
              16: 1975,
              17: 1975,
              18: 1975,
              19: 1975,
              20: 1975,
              21: 1975,
              22: 1975,
              23: 1975,
              24: 1975,
              25: 1975,
              26: 1975,
              27: 1975,
              28: 1975,
              29: 1975,
              30: 1975,
              31: 1975,
              32: 1975,
              33: 1975,
              34: 1975,
              35: 1975,
              36: 1975,
              37: 1975,
              38: 1975,
              39: 1975,
              40: 1975,
              41: 1975,
              42: 1975,
              43: 1975,
              44: 1975,
              45: 1975,
              46: 1975,
              47: 1975,
              48: 1975,
              49: 1975,
              50: 1975,
              51: 1975,
              52: 1975,
              53: 1975,
              54: 1975,
              55: 1975,
              56: 1975,
              57: 1975,
              58: 1975,
              59: 1975,
              60: 1975,
              61: 1975,
              62: 1975,
              63: 1975,
              64: 1975,
              65: 1975,
              66: 1975,
              67: 1975,
              68: 1975,
              69: 1975,
              70: 1975,
              71: 1975,
              72: 1975,
              73: 1975,
              74: 1975,
              75: 1975,
              76: 1975,
              77: 1975,
              78: 1975,
              79: 1975,
              80: 1975,
              81: 1975,
              82: 1975,
              83: 1975,
              84: 1975,
              85: 1975,
              86: 1975,
              87: 2012,
              88: 2012,
              89: 2012,
              90: 2012,
              91: 2012,
              92: 2012,
              93: 2012,
              94: 2012,
              95: 2012,
              96: 2012,
              97: 2012,
              98: 2012,
              99: 2012,
              100: 2012,
              101: 2012,
              102: 2012,
              103: 2012,
              104: 2012,
              105: 2012,
              106: 2012,
              107: 2012,
              108: 2012,
              109: 2012,
              110: 2012,
              111: 2012,
              112: 2012,
              113: 2012,
              114: 2012,
              115: 2012,
              116: 2012,
              117: 2012,
              118: 2012,
              119: 2012,
              120: 2012,
              121: 2012,
              122: 2012,
              123: 2012,
              124: 2012,
              125: 2012,
              126: 2012,
              127: 2012,
              128: 2012,
              129: 2012,
              130: 2012,
              131: 2012,
              132: 2012,
              133: 2012,
              134: 2012,
              135: 2012,
              136: 2012,
              137: 2012,
              138: 2012,
              139: 2012,
              140: 2012,
              141: 2012,
              142: 2012,
              143: 2012,
              144: 2012,
              145: 2012,
              146: 2012,
              147: 2012,
              148: 2012,
              149: 2012,
              150: 2012,
              151: 2012,
              152: 2012,
              153: 2012,
              154: 2012,
              155: 2012,
              156: 2012,
              157: 2012,
              158: 2012,
              159: 2012,
              160: 2012,
              161: 2012,
              162: 2012,
              163: 2012,
              164: 2012,
              165: 2012,
              166: 2012,
              167: 2012,
              168: 2012,
              169: 2012,
              170: 2012,
              171: 2012,
              172: 2012,
              173: 2012,
              174: 2012,
              175: 2012,
              176: 2012,
              177: 2012,
              178: 2012,
              179: 2012,
              180: 2012,
              181: 2012,
              182: 2012,
              183: 2012,
              184: 2012,
              185: 2012,
              186: 2012,
              187: 2012,
              188: 2012,
              189: 2012,
              190: 2012,
              191: 2012,
              192: 2012,
              193: 2012,
              194: 2012,
              195: 2012,
              196: 2012,
              197: 2012,
              198: 2012,
              199: 2012,
              200: 2012,
              201: 2012,
              202: 2012,
              203: 2012,
              204: 2012,
              205: 2012,
              206: 2012,
              207: 2012,
              208: 2012,
              209: 2012,
              210: 2012,
              211: 2012,
              212: 2012,
              213: 2012}})

bl_1975 = np.array([13.9, 14., 12.9, 13.5, 12.9, 14.6, 13., 14.2, 14.,
                    14.2, 13.1, 15.1, 13.5, 14.4, 14.9, 12.9, 13., 14.9,
                    14., 13.8, 13., 14.75, 13.7, 13.8, 14., 14.6, 15.2,
                    13.5, 15.1, 15., 12.8, 14.9, 15.3, 13.4, 14.2, 15.1,
                    15.1, 14., 13.6, 14., 14., 13.9, 14., 14.9, 15.6,
                    13.8, 14.4, 12.8, 14.2, 13.4, 14., 14.8, 14.2, 13.5,
                    13.4, 14.6, 13.5, 13.7, 13.9, 13.1, 13.4, 13.8, 13.6,
                    14., 13.5, 12.8, 14., 13.4, 14.9, 15.54, 14.63, 14.73,
                    15.73, 14.83, 15.94, 15.14, 14.23, 14.15, 14.35, 14.95, 13.95,
                    14.05, 14.55, 14.05, 14.45, 15.05, 13.25])

bl_2012 = np.array([14.3, 12.5, 13.7, 13.8, 12., 13., 13., 13.6, 12.8,
                    13.6, 12.95, 13.1, 13.4, 13.9, 12.3, 14., 12.5, 12.3,
                    13.9, 13.1, 12.5, 13.9, 13.7, 12., 14.4, 13.5, 13.8,
                    13., 14.9, 12.5, 12.3, 12.8, 13.4, 13.8, 13.5, 13.5,
                    13.4, 12.3, 14.35, 13.2, 13.8, 14.6, 14.3, 13.8, 13.6,
                    12.9, 13., 13.5, 13.2, 13.7, 13.1, 13.2, 12.6, 13.,
                    13.9, 13.2, 15., 13.37, 11.4, 13.8, 13., 13., 13.1,
                    12.8, 13.3, 13.5, 12.4, 13.1, 14., 13.5, 11.8, 13.7,
                    13.2, 12.2, 13., 13.1, 14.7, 13.7, 13.5, 13.3, 14.1,
                    12.5, 13.7, 14.6, 14.1, 12.9, 13.9, 13.4, 13., 12.7,
                    12.1, 14., 14.9, 13.9, 12.9, 14.6, 14., 13., 12.7,
                    14., 14.1, 14.1, 13., 13.5, 13.4, 13.9, 13.1, 12.9,
                    14., 14., 14.1, 14.7, 13.4, 13.8, 13.4, 13.8, 12.4,
                    14.1, 12.9, 13.9, 14.3, 13.2, 14.2, 13., 14.6, 13.1,
                    15.2])




"""

  Tasks 

"""


def do_pairs_regression(bl_1975, bd_1975, bl_2012, bd_2012):
    # Compute the ACTUAL linear regressions
    slope_1975, intercept_1975, *junk = np.polyfit(bl_1975, bd_1975, deg=1)
    slope_2012, intercept_2012, *junk = np.polyfit(bl_2012, bd_2012, deg=1)

    # Perform pairs BOOTSTRAPS for the linear regressions
    bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(bl_1975, bd_1975, size=1000)
    bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(bl_2012, bd_2012, size=1000)

    # Compute confidence intervals of slopes
    slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
    slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])

    intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])
    intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])

    # Print the results
    print('1975: slope =', slope_1975,
          'conf int =', slope_conf_int_1975)
    print('1975: intercept =', intercept_1975,
          'conf int =', intercept_conf_int_1975)
    print('2012: slope =', slope_2012,
          'conf int =', slope_conf_int_2012)
    print('2012: intercept =', intercept_2012,
          'conf int =', intercept_conf_int_2012)

    plot_linreg_bootstraps(bl_1975, bd_1975, bl_2012, bd_2012, bs_slope_reps_1975, bs_intercept_reps_1975, bs_slope_reps_2012, bs_intercept_reps_2012)


def plot_linreg_bootstraps(bl_1975, bd_1975, bl_2012, bd_2012, bs_slope_reps_1975, bs_intercept_reps_1975, bs_slope_reps_2012, bs_intercept_reps_2012):
    # Make scatter plot of 1975 data
    _ = plt.plot(bl_1975, bd_1975, marker='.',
                 linestyle='none', color='blue', alpha=0.5)

    # Make scatter plot of 2012 data
    _ = plt.plot(bl_2012, bd_2012, marker='.',
                 linestyle='none', color='red', alpha=0.5)

    # Label axes and make legend
    _ = plt.xlabel('beak length (mm)')
    _ = plt.ylabel('beak depth (mm)')
    _ = plt.legend(('1975', '2012'), loc='upper left')

    # Generate x-values for bootstrap lines: x
    x = np.array([10, 17])

    # Plot the bootstrap lines
    for i in range(100):
        """
            bs_slope_reps_1975[i]*x  -- creates a two column array using x - this is a two point line with starting at 0 and ending on 100 * the slope
           bs_slope_reps_1975[i]*x + bs_intercept_reps_1975[i]   -- multiplies the values in the two point line by th intercept pushing the line higher ot lower
        """

        plt.plot(x, bs_slope_reps_1975[i]*x + bs_intercept_reps_1975[i],
                 linewidth=0.5, alpha=0.2, color='blue')
        plt.plot(x, bs_slope_reps_2012[i]*x + bs_intercept_reps_2012[i],
                 linewidth=0.5, alpha=0.2, color='red')

    # Draw the plot again
    plt.show()


def beak_length_by_depth(bl_1975, bd_1975, bl_2012, bd_2012):
    # Compute length-to-depth ratios
    ratio_1975 = bl_1975 / bd_1975
    ratio_2012 = bl_2012 / bd_2012

    # Compute means
    mean_ratio_1975 = np.mean(ratio_1975)
    mean_ratio_2012 = np.mean(ratio_2012)

    # Generate bootstrap replicates of the means
    bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, size=10000)
    bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, size=10000)

    # Compute the 99% confidence intervals
    conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
    conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

    # Print the results
    print('1975: mean ratio =', mean_ratio_1975,
          'conf int =', conf_int_1975)
    print('2012: mean ratio =', mean_ratio_2012,
          'conf int =', conf_int_2012)


def eda_beak_length_depth(bl_1975, bd_1975, bl_2012, bd_2012):
    # Make scatter plot of 1975 data
    _ = plt.plot(bl_1975, bd_1975, marker='.',
                 linestyle='None', color='blue', alpha=0.5)

    # Make scatter plot of 2012 data
    _ = plt.plot(bl_2012, bd_2012, marker='.',
                 linestyle='None', color='red', alpha=0.5)

    # Label axes and make legend
    _ = plt.xlabel('beak length (mm)')
    _ = plt.ylabel('beak depth (mm)')
    _ = plt.legend(('1975', '2012'), loc='upper left')

    # Show the plot
    plt.show()


def plot_beeswarm(df):
    # Create bee swarm plot
    ax = sns.swarmplot(x="year", y="beak_depth", data=df)

    # Label the axes
    _ = plt.xlabel('year')
    _ = plt.ylabel('beak depth (mm)')

    # Show the plot
    plt.show()


def plot_ecdfs(bd_1975, bd_2012):
    # Compute ECDFs
    x_1975, y_1975 = ecdf(bd_1975)
    x_2012, y_2012 = ecdf(bd_2012)

    # Plot the ECDFs
    _ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
    _ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

    # Set margins
    plt.margins(.02)

    # Add axis labels and legend
    _ = plt.xlabel('beak depth (mm) sorted')
    _ = plt.ylabel('ECDF - N-normalized')
    _ = plt.legend(('1975', '2012'), loc='lower right')

    # Show the plot
    plt.show()


def diff_sample_means(bd_1975, bd_2012):
    # Compute the difference of the sample means: mean_diff
    mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

    # Get bootstrap replicates of means
    bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
    bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

    # Compute samples of difference of means: bs_diff_replicates
    bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

    # Compute 95% confidence interval: conf_int
    conf_int = (np.percentile(bs_diff_replicates, 2.5), np.percentile(bs_diff_replicates, 97.5))

    # Print the results
    print('difference of means =', mean_diff, 'mm')
    print('95% confidence interval =', conf_int, 'mm')


def calc_hypothesis(bd_1975, bd_2012, mean_diff):
    # Compute mean of combined data set: combined_mean
    combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

    # Shift the samples
    bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
    bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

    # Get bootstrap replicates of shifted data sets
    bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, size=10000)
    bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, size=10000)

    # Compute replicates of difference of means: bs_diff_replicates
    bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

    # Compute the p-value
    p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

    # Print p-value
    print('p =', p)


bd_1975 = df.loc[(df.year == 1975), 'beak_depth'].to_numpy()
bd_2012 = df.loc[(df.year == 2012), 'beak_depth'].to_numpy()
mean_diff = 0.22622047244094645

# plot_beeswarm(df=df)
# plot_ecdfs(bd_1975, bd_2012)
# diff_sample_means(bd_1975, bd_2012)
# calc_hypothesis(bd_1975, bd_2012, mean_diff)
eda_beak_length_depth(bd_1975, bl_1975, bd_2012, bl_2012)
beak_length_by_depth(bd_1975, bl_1975, bd_2012, bl_2012)
# do_pairs_regression(bl_1975, bd_1975, bl_2012, bd_2012)


#%%
