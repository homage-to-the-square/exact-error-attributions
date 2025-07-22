import numpy as np
import pandas as pd
from itertools import permutations
from scipy.stats import norm

"""
    Simple helper for binomial data:
"""
def is_binomial_data_seperable(
    df: pd.DataFrame, binary_col: str, cont_col: str
) -> bool:
    """
    Given a dataframe, column for binary rv, and column for continuous rv, returns True if the
        binary rv is seperable.
    """
    separability_check_df = (
        df[[binary_col, cont_col]].groupby(binary_col)[cont_col].agg(["min", "max"])
    )

    # checks if there is y = 0, y = 1, as well as whether min(y = 0) > max(y = 1), and vice versa
    return (
        (len(separability_check_df) < 2)
        or (separability_check_df.iloc[0, 0] > separability_check_df.iloc[1, 1])
        or (separability_check_df.iloc[1, 0] > separability_check_df.iloc[0, 1])
    )
    
"""
    note that for the cloglog link $g(\mu) = \log(-\log(1-\mu))$, its derivative is $-1/(\log(1-x)(1-x))$, while for the probit link, it is simply $1/\phi(\Phi^{-1}(\mu))$.
"""
def compute_mu_i(X: np.array, beta: np.array, model_type = 'Logit'):
    # X can be 2d or 1d
    
    if np.sum(~(beta == 0)) == 0:
        return 1/2
    
    if model_type == 'Logit':
        return 1/(1 + np.exp(- X @ beta))
    elif model_type == 'Probit':
        return norm.cdf(X @ beta)
    elif model_type == 'CLogLog':
        return 1 - np.exp(-np.exp(X @ beta))
    else:
        return None


def compute_sum_G_i(x, y, beta, model_type = 'Logit'):
    # here we assume x is 2d.
    # print(compute_mu_i(x, beta, model_type = model_type))

    if model_type == 'Logit':
        return x.T @ (y - compute_mu_i(x, beta, model_type = model_type))
    else:
        temp_mu_is = compute_mu_i(x, beta, model_type = model_type)
        variance_denominator = 1/((temp_mu_is) * (1 - temp_mu_is))
        
        if model_type == 'Probit':
            dmu_dg = norm.pdf(x @ beta)
        elif model_type == 'CLogLog':
            dmu_dg = np.log(1 - temp_mu_is) * (temp_mu_is - 1)
        else:
            return None
        
        return np.sum(x.mul(((y - temp_mu_is) * variance_denominator * dmu_dg), axis=0))

"""
    Utils to compute Jn, the sample elasticity
"""

def generate_all_paths(number_of_coefficients):
    # simply returns a list of tuples of paths from a to b
    return list(permutations(range(number_of_coefficients)))

def generate_all_beta_pairs_per_row(path, pop_beta, sample_beta):
    ### given a path (tuple), retuns a list of pairs of betas [(0, a, b), (1, c, d), ...] 
    ###### where the first pair is for the first column, ...
    
    assert len(path) == len(pop_beta)
    assert len(pop_beta) == len(sample_beta)
    
    prev_beta = pop_beta.copy()
    
    list_of_betas = []
    
    for ele in path:
        new_beta = prev_beta.copy()
        new_beta[ele] = sample_beta[ele]
        list_of_betas.append([ele, prev_beta, new_beta])
        
        prev_beta = new_beta.copy()
    
    list_of_betas.sort()
    
    return list_of_betas

def compute_path_specific_jn(path, pop_beta, sample_beta, x, y, model_type = 'Logit'):
    # given a specific path, computes the Jn matrix.
    path_betas = generate_all_beta_pairs_per_row(path, pop_beta, sample_beta)
    num_coef = len(pop_beta)
    num_obs = len(x)
    unnormalized_jn = pd.concat([(compute_sum_G_i(x, y, path_betas[temp_col][2], model_type = model_type) \
                                     - compute_sum_G_i(x, y, path_betas[temp_col][1], model_type = model_type)) / \
                                    (sample_beta[temp_col] - pop_beta[temp_col]) \
                                 for temp_col in range(num_coef)], axis=1)
    
    return -unnormalized_jn / num_obs

def compute_average_jn(pop_beta, sample_beta, x, y, model_type = 'Logit'):
    num_obs = len(pop_beta)
    all_paths = generate_all_paths(num_obs)
    
    all_path_specific_jns = [compute_path_specific_jn(temp_path, pop_beta, \
                                                      sample_beta, x, y, model_type = model_type) for temp_path in all_paths]
    
    return pd.concat(all_path_specific_jns).reset_index().groupby('index').mean()
