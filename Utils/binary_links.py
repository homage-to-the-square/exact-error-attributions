import numpy as np
from scipy.special import expit
from scipy.stats import norm


class Link_Function_For_Binary_Response:
    """
    General class for link functions for binary response models.
        - Given the quantile function (e.g., inverse link function)
            and its derivatives, defines both the functions to 
            compute the observed Fisher Information + its derivative w.r.t. beta.
        - For derivations of fi and fi_deriv, see binary_links.md
    """

    def quantile(self, x, b):
        raise NotImplementedError

    def derivative_quantile(self, x, b):
        """
        The derivative of the quantile function, w.r.t. eta = x * beta.
        """
        raise NotImplementedError

    def second_derivative_quantile(self, x, b):
        raise NotImplementedError

    def third_derivative_quantile(self, x, b):
        raise NotImplementedError

    def fi(self, x, b, y):
        assert (y == 0) | (y == 1)

        inv = self.quantile(x, b)
        first_d = self.derivative_quantile(x, b)
        second_d = self.second_derivative_quantile(x, b)

        return (x**2) * (1 - y) * (second_d * (1 - inv) + (first_d**2)) / (
            (1 - inv) ** 2
        ) - (x**2) * y * (second_d * inv - (first_d**2)) / (inv**2)

    def fi_deriv(self, x, b, y):
        # this is the observed fisher information calculation

        assert (y == 0) | (y == 1)

        inv = self.quantile(x, b)
        first_d = self.derivative_quantile(x, b)
        second_d = self.second_derivative_quantile(x, b)
        third_d = self.third_derivative_quantile(x, b)

        # first compute the numerators
        first_half_numerator = (
            third_d * ((1 - inv) ** 2)
            + 3 * second_d * first_d * (1 - inv)
            + 2 * (first_d**3)
        )

        second_half_numerator = (
            third_d * (inv**2) - 3 * second_d * first_d * inv + 2 * (first_d**3)
        )

        return (x**3) * (1 - y) * (first_half_numerator) / (
            (1 - inv) ** 3
        ) - x**3 * y * second_half_numerator / (inv**3)


"""
    We inherit the class above for the three link functions:
"""
class Logit_Link(Link_Function_For_Binary_Response):
    def quantile(self, x, b):
        return expit(x * b)

    def derivative_quantile(self, x, b):
        inv = self.quantile(x, b)
        return inv * (1 - inv)

    def second_derivative_quantile(self, x, b):
        inv = self.quantile(x, b)
        return (inv) * (1 - inv) * (1 - 2 * inv)

    def third_derivative_quantile(self, x, b):
        temp_exp = np.exp(x * b)
        return (temp_exp) * (temp_exp**2 - 4 * temp_exp + 1) / ((temp_exp + 1) ** 4)


class Probit_Link(Link_Function_For_Binary_Response):
    def quantile(self, x, b):
        return norm.cdf(x * b)

    def derivative_quantile(self, x, b):
        return norm.pdf(x * b)

    def second_derivative_quantile(self, x, b):
        eta = x * b
        return (-eta) * norm.pdf(eta)

    def third_derivative_quantile(self, x, b):
        eta = x * b
        return (eta**2 - 1) * norm.pdf(eta)


class CLogLog_Link(Link_Function_For_Binary_Response):
    def quantile(self, x, b):
        return 1 - np.exp(-np.exp(x * b))

    def derivative_quantile(self, x, b):
        eta = x * b
        return np.exp(eta - np.exp(eta))

    def second_derivative_quantile(self, x, b):
        eta = x * b
        return (1 - np.exp(eta)) * np.exp(eta - np.exp(eta))

    def third_derivative_quantile(self, x, b):
        eta = x * b
        return (np.exp(2 * eta) - 3 * np.exp(eta) + 1) * np.exp(
            eta - np.exp(eta)
        )