import numpy as np
import pandas as pd
from itertools import permutations
from scipy.stats import norm
import statsmodels.api as sm
from typing import Union

"""
    Simple helpers, for 4.2:
"""
def compute_all_three_logistic_models(x_data, y_data):
    """
        given (x_data, y_data), returns the logit, probit, and cloglog models given the fit
            without an intercept.
    """
    temp_logit_model = sm.Logit(endog=y_data, exog=x_data).fit(disp=0)
    temp_probit_model = sm.GLM(
        y_data, x_data, family=sm.families.Binomial(link=sm.families.links.Probit())
    ).fit()
    temp_cloglog_model = sm.GLM(
        y_data, x_data, family=sm.families.Binomial(link=sm.families.links.CLogLog())
    ).fit()
    return [temp_logit_model, temp_probit_model, temp_cloglog_model]

def get_pop_gs_for_binary_y(population_models, pop_x, pop_y, population_size) -> dict:
    pop_gs = {}
    pop_logit_model, pop_probit_model, pop_cloglog_model = population_models
    
    # compute logit gs:
    pop_gs["Logit"] = pop_x * (
        np.array(pop_y).reshape((population_size, 1))
        - np.array(pop_logit_model.predict()).reshape((population_size, 1))
    )

    # compute probit gs:
    temp_mu_is = pop_probit_model.predict()
    variance_denominator = 1/((temp_mu_is) * (1 - temp_mu_is))
    dmu_dg = norm.pdf(pop_x @ pop_probit_model.params)
    pop_gs['Probit'] = pop_x.mul(((pop_y - temp_mu_is) * variance_denominator * dmu_dg), axis=0)
    
    # compute cloglog gs:
    temp_mu_is = pop_cloglog_model.predict()
    variance_denominator = 1/((temp_mu_is) * (1 - temp_mu_is))
    dmu_dg = np.log(1 - temp_mu_is) * (temp_mu_is - 1)
    pop_gs['CLogLog'] = pop_x.mul(((pop_y - temp_mu_is) * variance_denominator * dmu_dg), axis=0)

    return pop_gs

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
    Helpers to compute the Jn matrix:
"""
def compute_mu_i(X: np.array, beta: np.array, link_fn: str = 'Logit'):
    """
        Computes the mu_i of a binary regression, via the inverse link function.
        - X can be 1d or 2d
        - but, beta cannot be a float, because of the matrix multiplication below.
    """
    
    if np.sum(~(beta == 0)) == 0:
        return 1/2

    eta = X @ beta
    if link_fn == 'Logit':
        return 1/(1 + np.exp(- eta))
    elif link_fn == 'Probit':
        return norm.cdf(eta)
    elif link_fn == 'CLogLog':
        return 1 - np.exp(-np.exp(eta))
    else:
        return None


def compute_sum_G_i(
    X: Union[pd.Series, pd.DataFrame],
    y: Union[pd.Series, pd.DataFrame],
    beta,
    link_fn: str = "Logit",
) -> pd.Series:
    """
        Here we assume x is 2d. To see the derivations of dmu_dg, see ddc_utils.md
    """

    if link_fn == "Logit":
        return X.T @ (y - compute_mu_i(X, beta, link_fn=link_fn))
    else:
        temp_mu_is = compute_mu_i(X, beta, link_fn=link_fn)
        variance_denominator = 1 / ((temp_mu_is) * (1 - temp_mu_is))

        if link_fn == "Probit":
            dmu_dg = norm.pdf(X @ beta)
        elif link_fn == "CLogLog":
            dmu_dg = np.log(1 - temp_mu_is) * (temp_mu_is - 1)
        else:
            return None

        return np.sum(X.mul((y - temp_mu_is) * variance_denominator * dmu_dg, axis=0), axis=0)


"""
    Utils to compute Jn, the sample elasticity
"""

def generate_all_paths(number_of_coefficients: int) -> list:
    """
        simply returns a list of tuples of permutations of num_coef
        - each permutation is interpreted as a path from one vertex to its diagonal vertex in the hypercube
            by reading the permutation from left to right, and switching that coordinate in the permutation.

        Ex:
            For permutation (2, 0, 1), if going from a = (a1, a2, a3) to b = (b1, b2, b3),
                the path implied is:
                    (a1, a2, a3) -> (a1, a2, b3) -> (b1, a2, b3) -> (b1, b2, b3).
    """
    return list(permutations(range(number_of_coefficients)))

def generate_all_steps_for_path(path: tuple, pop_beta, sample_beta) -> list:
    """
        Given a path (tuple) from the pop_beta to the sample_beta,
            returns a list of the path's steps in terms of betas,
            ordered such that the nth element is the step taken when switching the nth coordinate.

        Continuing the example of permutation (2, 0, 1), if going from a = (a1, a2, a3) to b = (b1, b2, b3),
            the following code below will return:
                [
                    (0, (a1, a2, b3), (b1, a2, b3)),
                    (1, (b1, a2, b3), (b1, b2, b3)),
                    (2, (a1, a2, a3), (a1, a2, b3))
                ]
    """
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

def compute_path_specific_jn(
    path,
    pop_beta,
    sample_beta,
    X: Union[pd.Series, pd.DataFrame],
    y: Union[pd.Series, pd.DataFrame],
    link_fn: str = "Logit",
):
    """
    given a specific path, computes the Jn^s matrix for that path
    """
    path_betas = generate_all_steps_for_path(path, pop_beta, sample_beta)
    num_coef = len(pop_beta)
    num_obs = len(X)
    unnormalized_jn = pd.concat(
        [
            (
                compute_sum_G_i(X, y, path_betas[temp_col][2], link_fn=link_fn)
                - compute_sum_G_i(X, y, path_betas[temp_col][1], link_fn=link_fn)
            )
            / (sample_beta[temp_col] - pop_beta[temp_col])
            for temp_col in range(num_coef)
        ],
        axis=1,
    )

    return -unnormalized_jn / num_obs

def compute_average_jn(
    pop_beta,
    sample_beta,
    X: Union[pd.Series, pd.DataFrame],
    y: Union[pd.Series, pd.DataFrame],
    link_fn: str = "Logit",
):
    """
    Note that x, y here are assumed to be of the sample x, ys.
    - Note: X, y must be pd.Series/pd.DataFrame, because internally, a pd.concat is being used.
    """
    num_coefs = len(pop_beta)
    all_paths = generate_all_paths(num_coefs)

    all_path_specific_jns = [
        compute_path_specific_jn(
            temp_path, pop_beta, sample_beta, X, y, link_fn=link_fn
        )
        for temp_path in all_paths
    ]

    return pd.concat(all_path_specific_jns).reset_index().groupby("index").mean()
