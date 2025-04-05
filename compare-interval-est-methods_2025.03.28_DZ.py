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

true_proportion = [0.1, 0.3, 0.5, 0.7, 0.9]

# Number of simulations
n_simulations = 1000


#---------------------------------------------------------------------
# Implement Wald, Wilson, and Jeffreys methods for interval estimation
#----------------------------------------------------------------------

cases = 1
n = 100

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

# Plot coverage probabilities
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(data=results, x='sample_size', y='coverage_probability', hue='method')
plt.title('Coverage Probability of Different Methods')
plt.xlabel('Sample Size')
plt.ylabel('Coverage Probability')
plt.legend(title='Method')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('coverage_probability.png')
plt.show()


#-----------------------------------------------------------------------
# b.Average length of confidence intervals: the mean width of the CIs, indicating precision.
#-----------------------------------------------------------------------

results_precision = pd.DataFrame(columns=['sample_size', 'true_proportion', 'method', 'average_length'])

# Initialize counts
length_wald = []
length_wilson = []
length_jeffreys = []

for n in sample_sizes:
    for p in true_proportion:
        # Generate binomial data
        simulated_data = np.random.binomial(n, p, n_simulations)

        for i in range(n_simulations):
            count = simulated_data[i]

        #     # Wald method
        #     ci_wald = proportion_confint(count, n, alpha=0.05, method='normal')
        #     if ci_wald[0] <= p <= ci_wald[1]:
        #         coverage_wald += 1

        #     # Wilson method
        #     ci_wilson = proportion_confint(count, n, alpha=0.05, method='wilson')
        #     if ci_wilson[0] <= p <= ci_wilson[1]:
        #         coverage_wilson += 1
            
        #     # Jeffrey method
        #     ci_jeffreys = proportion_confint(count, n, alpha=0.05, method='jeffreys')
        #     if ci_jeffreys[0] <= p <= ci_jeffreys[1]:
        #         coverage_jeffreys += 1

        # # Calculate coverage probabilities
        # coverage_probability_wald = coverage_wald / n_simulations
        # coverage_probability_wilson = coverage_wilson / n_simulations
        # coverage_probability_jeffreys = coverage_jeffreys / n_simulations





#--------------------------------------------
# Visualize the results 
#--------------------------------------------

