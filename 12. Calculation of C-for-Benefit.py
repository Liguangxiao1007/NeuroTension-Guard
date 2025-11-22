
## Annotated Version
import random  # Import random module for random sampling
import numpy as np  # Import NumPy library for numerical computations
import pandas as pd  # Import Pandas library for data processing
import time  # Import time module for timing operations
from joblib import Parallel, delayed  # Import parallel processing functions from Joblib
import multiprocessing  # Import multiprocessing module for parallel operations

# Input file requirements:
# The CSV file should contain the following columns:
# - 'preds': predicted risk reductions (continuous values)
# - 'outcome': binary outcomes (0 or 1)
# - 'treated': treatment indicators (0 for control, 1 for treated)

def c_statistic(pred_rr, y, w):
  """
    Calculate concordance-for-benefit (C-for-benefit) statistic.
    
    Parameters:
    pred_rr: array-like, predicted risk reductions
    y: array-like, binary outcomes (0 or 1)
    w: array-like, treatment indicators (0 for control, 1 for treated)
    
    Returns:
    float: C-for-benefit statistic value
    """

# Create tuples of predictions, outcomes, and treatment indicators
tuples = list(zip(pred_rr, y, w))

# Separate treated and untreated groups
untreated = [t for t in tuples if t[2] == 0]  # Control group
treated = [t for t in tuples if t[2] == 1]    # Treatment group

# Balance group sizes by random sampling
if len(treated) < len(untreated):
  untreated = random.sample(untreated, len(treated))
if len(untreated) < len(treated):
  treated = random.sample(treated, len(untreated))
assert len(untreated) == len(treated)  # Verify equal group sizes

# Sort groups by predicted risk reduction
untreated = sorted(untreated, key=lambda t: t[0])
treated = sorted(treated, key=lambda t: t[0])

# Define observed benefit scoring dictionary
obs_benefit_dict = {
  (0, 0): 0,   # No event in either group: no benefit
  (0, 1): -1,  # Event in treated only: harm
  (1, 0): 1,   # Event in control only: benefit
  (1, 1): 0,   # Event in both groups: no benefit
}

# Pair treated and untreated subjects
pairs = list(zip(untreated, treated))

# Calculate observed and predicted benefits for each pair
obs_benefit = [obs_benefit_dict[(t[1], u[1])] for (u, t) in pairs]
pred_benefit = [np.mean([t[0], u[0]]) for (u, t) in pairs]

# Calculate concordance
count, total = 0, 0
for i in range(len(pairs)):
  for j in range(i + 1, len(pairs)):
  if obs_benefit[i] != obs_benefit[j]:
  if (obs_benefit[i] < obs_benefit[j] and pred_benefit[i] < pred_benefit[j]) or \
(obs_benefit[i] > obs_benefit[j] and pred_benefit[i] > pred_benefit[j]):
  count += 1  # Concordant pair
total += 1  # Total comparable pairs

return count / total if total > 0 else 0  # Return C-for-benefit statistic

def bootstrap_sample(data, seed):
  """
    Perform bootstrap sampling and calculate C-for-benefit.
    
    Parameters:
    data: DataFrame containing predictions, outcomes, and treatment indicators
    seed: random seed for reproducibility
    
    Returns:
    float: C-for-benefit statistic for bootstrap sample
    """

if seed is not None:
  random.seed(seed)
# Generate bootstrap sample with replacement
samp = data.sample(n=len(data), replace=True, random_state=seed)

# Calculate C-for-benefit for bootstrap sample
result = c_statistic(
  pred_rr=samp['preds'],
  y=samp['outcome'],
  w=samp['treated']
)
return result

def c4benefit_bootstrap_func(data, n_bootstraps=1000, seed=1, n_jobs=None):
  """
    Calculate C-for-benefit with 95% confidence intervals using bootstrap.
    
    Parameters:
    data: DataFrame containing the required columns
    n_bootstraps: number of bootstrap iterations
    seed: random seed for reproducibility
    n_jobs: number of parallel jobs (-1 for all available cores)
    
    Returns:
    DataFrame: C-for-benefit estimate with confidence intervals
    """

# Calculate C-for-benefit on original data
c4ben_est = c_statistic(
  pred_rr=data['preds'],
  y=data['outcome'],
  w=data['treated']
)

# Generate random seeds for bootstrap iterations
random.seed(seed)
seeds = [seed + i for i in range(n_bootstraps)]

# Perform bootstrap in parallel
c4ben_data = Parallel(n_jobs=n_jobs)(
  delayed(bootstrap_sample)(data, s) for s in seeds
)

# Calculate 95% confidence intervals
c4ben_ci = np.percentile(c4ben_data, [2.5, 97.5])

# Create results DataFrame
result = pd.DataFrame({
  'c4ben_est': [c4ben_est],
  'c4ben_lower': [c4ben_ci[0]],
  'c4ben_upper': [c4ben_ci[1]]
})

return result

# Load data from CSV file
# File should contain columns: preds, outcome, treated
data = pd.read_csv('./data/c_for_benefit_analysis_data.csv')

# Estimate computation time
start_time = time.time()

# Calculate C-for-benefit with bootstrap confidence intervals
# n_jobs parameter explanation:
# - None: use 1 core (sequential processing)
# - -1: use all available CPU cores
# - Positive integer: use specified number of cores
c_for_ben = c4benefit_bootstrap_func(
  data, 
  n_bootstraps=1000, 
  seed=1,  # Set seed to 1 for reproducibility
  n_jobs=-1  # Use all available cores
)

end_time = time.time()

# Display results
print("C-for-Benefit Results:")
print(c_for_ben)
print(f"Computation time: {end_time - start_time:.2f} seconds")