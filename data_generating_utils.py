import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


# generate population data:
def generate_population_data(
    population_size,
    number_of_coefficients,
    rand_generator,
    feature_cols,
    true_beta=0,
    link="Logit",
):
    """
    Generate simulated data for logistic regression analysis.

    """
    population_x = rand_generator.multivariate_normal(
        mean=np.zeros(number_of_coefficients),
        cov=np.eye(number_of_coefficients),
        size=population_size,
    )

    if true_beta != 0:
        eta = true_beta * population_x
        if link == "Logit":
            population_y = rand_generator.binomial(n=1, p=expit(eta)).flatten()
        elif link == "Probit":
            population_y = rand_generator.binomial(
                n=1, p=norm.cdf(eta)
            ).flatten()
        elif link == "CLogLog":
            population_y = rand_generator.binomial(
                n=1, p=(1 - np.exp(-np.exp(eta)))
            ).flatten()
        else:
            return pd.DataFrame()
    else:
        population_y = rand_generator.binomial(n=1, p=1 / 2, size=population_size)

    population_data = pd.concat(
        [pd.Series(population_y), pd.DataFrame(population_x)], axis=1
    )

    population_data.columns = ["y"] + feature_cols

    return population_data