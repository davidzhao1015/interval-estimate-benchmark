#=============================================================================== 
# This script compares the performance of different interval estimation methods
#================================================================================

#------------------------------------------
# Import necessary libraries
#------------------------------------------


#-------------------------------------------
# Generate synthetic data
#-------------------------------------------

from statsmodels.stats.proportion import proportion_confint

# Set the random seed for reproducibility
import numpy as np
np.random.seed(123)

# Define the parameters for the binomial distribution
sample_sizes = [30, 100, 500]

# true_proportion = [0.1, 0.3, 0.5, 0.7, 0.9]
true_proportion = np.linspace(0.001, 0.1, 5)

# Number of simulations
n_simulations = 1000


#---------------------------------------------------------------------
# Implement Wald, Wilson, and Jeffreys methods for interval estimation
#----------------------------------------------------------------------

cases = 1
n = 2000

# 1. Wald method
ci_wald_estimate = proportion_confint(cases, n, alpha=0.05, method='normal')
print(f'95% Wald CI: [{ci_wald_estimate[0]:.4f}, {ci_wald_estimate[1]:.4f}]') 

# 2. Wilson method
ci_wilson_estimate = proportion_confint(cases, n, alpha=0.05, method='wilson')
print(f'95% Wilson CI: [{ci_wilson_estimate[0]:.4f}, {ci_wilson_estimate[1]:.4f}]')

# 3. Jeffreys method
ci_jeffreys_estimate = proportion_confint(cases, n, alpha=0.05, method='jeffreys')
print(f'95% Jeffreys CI: [{ci_jeffreys_estimate[0]:.4f}, {ci_jeffreys_estimate[1]:.4f}]')




#-----------------------------------------------------------------------
# a. Coverage probability:The proportion of times the true parameter lies within the confidence interval
#-----------------------------------------------------------------------

# Initialize dataframe to store results
import pandas as pd

results = pd.DataFrame(columns=['sample_size', 'true_proportion', 'method', 'coverage_probability'])

# Initialize counts
coverage_wald = 0
coverage_wilson = 0
coverage_jeffreys = 0

for n in sample_sizes:
    for p in true_proportion:
        # Generate binomial data
        simulated_data = np.random.binomial(n, p, n_simulations)

        for i in range(n_simulations):
            count = simulated_data[i]

            # Wald method
            ci_wald = proportion_confint(count, n, alpha=0.05, method='normal')
            if ci_wald[0] <= p <= ci_wald[1]:
                coverage_wald += 1

            # Wilson method
            ci_wilson = proportion_confint(count, n, alpha=0.05, method='wilson')
            if ci_wilson[0] <= p <= ci_wilson[1]:
                coverage_wilson += 1
            
            # Jeffrey method
            ci_jeffreys = proportion_confint(count, n, alpha=0.05, method='jeffreys')
            if ci_jeffreys[0] <= p <= ci_jeffreys[1]:
                coverage_jeffreys += 1

        # Calculate coverage probabilities
        coverage_probability_wald = coverage_wald / n_simulations
        coverage_probability_wilson = coverage_wilson / n_simulations
        coverage_probability_jeffreys = coverage_jeffreys / n_simulations

        # Append results to the dataframe
        results = pd.concat([results, pd.DataFrame([{'sample_size': n, 'true_proportion': p, 
                                             'method': 'Wald', 
                                             'coverage_probability': coverage_probability_wald}])], 
                    ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'sample_size': n, 'true_proportion': p, 
                                             'method': 'Wilson', 
                                             'coverage_probability': coverage_probability_wilson}])],
                    ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'sample_size': n, 'true_proportion': p, 
                                             'method': 'Jeffreys', 
                                             'coverage_probability': coverage_probability_jeffreys}])],
                    ignore_index=True)
        
        # Reset counts for the next iteration
        coverage_wald = 0
        coverage_wilson = 0
        coverage_jeffreys = 0

results.head()


#-----------------------------------------------------------------------
# b.Average length of confidence intervals: the mean width of the CIs, indicating precision.
#-----------------------------------------------------------------------

results_precision = pd.DataFrame(columns=['sample_size', 'true_proportion', 'method', 'average_length', 'sd_length'])

# Initialize counts
ci_length_wald = []
ci_length_wilson = []
ci_length_jeffreys = []

for n in sample_sizes:
    for p in true_proportion:
        # Generate binomial data
        simulated_data = np.random.binomial(n, p, n_simulations)

        for i in range(n_simulations):
            count = simulated_data[i]

            # Wald method
            ci_wald = proportion_confint(count, n, alpha=0.05, method='normal')
            ci_length_wald.append(ci_wald[1] - ci_wald[0])

            # Wilson method
            ci_wilson = proportion_confint(count, n, alpha=0.05, method='wilson')
            ci_length_wilson.append(ci_wilson[1] - ci_wilson[0])
            
            # Jeffrey method
            ci_jeffreys = proportion_confint(count, n, alpha=0.05, method='jeffreys')
            ci_length_jeffreys.append(ci_jeffreys[1] - ci_jeffreys[0])

        # Calculate average lengths
        average_length_wald = np.mean(ci_length_wald)
        average_length_wilson = np.mean(ci_length_wilson)
        average_length_jeffreys = np.mean(ci_length_jeffreys)

        # Calculate standard deviations
        sd_length_wald = np.std(ci_length_wald)
        sd_length_wilson = np.std(ci_length_wilson)
        sd_length_jeffreys = np.std(ci_length_jeffreys)

        # Append results to the dataframe
        results_precision = pd.concat([results_precision, pd.DataFrame([{'sample_size': n, 'true_proportion': p, 
                                             'method': 'Wald', 
                                             'average_length': average_length_wald,
                                             'sd_length': sd_length_wald}])], 
                    ignore_index=True)
        results_precision = pd.concat([results_precision, pd.DataFrame([{'sample_size': n, 'true_proportion': p, 
                                             'method': 'Wilson', 
                                             'average_length': average_length_wilson,
                                             'sd_length': sd_length_wilson}])],
                    ignore_index=True)
        results_precision = pd.concat([results_precision, pd.DataFrame([{'sample_size': n, 'true_proportion': p, 
                                             'method': 'Jeffreys', 
                                             'average_length': average_length_jeffreys,
                                             'sd_length': sd_length_jeffreys}])],
                    ignore_index=True)

        # Reset the length for the next iteration
        ci_length_wald = []
        ci_length_wilson = []
        ci_length_jeffreys = []

results_precision.head(20)


#--------------------------------------------
# Visualize the results 
#--------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns


# To illustrate coverage probabilities by methods, draw the bar plot grouped by method, split by sample size
barplot_coverage = sns.catplot(
    data=results,
    x='sample_size', 
    y='coverage_probability', 
    hue='method',
    col='true_proportion',
    kind='bar',
    # height=4,
    # aspect=0.8,
    height=5,
    aspect=1.2,
    palette='muted',
    ci=None)
barplot_coverage.set_axis_labels("Sample Size", "Coverage Probability")
barplot_coverage.set(ylim=(0.75, 1.05))

# Add a horizontal line at y = 0.95 to all facets
for ax in barplot_coverage.axes.flat:
    ax.axhline(0.95, ls='--', color='red', linewidth=1.2)  # dashed red line

plt.show()
barplot_coverage.savefig("barplot_coverage_facet.png", dpi=300, bbox_inches='tight')


# To illustrate precisions (95 CI width) by methods, draw the bar plot grouped by method, split by sample size
barplot_precision = sns.catplot(
    data=results_precision,
    x='sample_size', 
    y='average_length', 
    hue='method',
    col='true_proportion',
    kind='bar',
    # height=4,
    # aspect=0.8,
    palette='muted',
    ci=None)

barplot_precision.set_axis_labels("Sample Size", "Average Length")
plt.show()

barplot_precision.savefig("barplot_precision_facet.png", dpi=300, bbox_inches='tight')
