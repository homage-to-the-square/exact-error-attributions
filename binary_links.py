from scipy.stats import norm
from scipy.special import expit
import numpy as np

class Link_Function_For_Binary_Response():
    """
        General class for link functions for binary response models.
            - Given the quantile function and its derivatives, defines both the 
                functions to compute the Fisher Information + its derivative w.r.t. beta.
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
        
        return (x ** 2) * (1-y) * (second_d * (1 - inv) + (first_d ** 2)) / ((1 - inv) ** 2) - \
                    (x ** 2) * y * (second_d * inv - (first_d ** 2)) / (inv ** 2)
    
    def fi_deriv(self, x, b, y):
        # this is the observed fisher information calculation
        
        assert (y == 0) | (y == 1)
        
        inv = self.quantile(x, b)
        first_d = self.derivative_quantile(x, b)
        second_d = self.second_derivative_quantile(x, b)
        third_d = self.third_derivative_quantile(x, b)
        
        # first compute the numerators
        first_half_numerator = (
            third_d * ((1 - inv)**2)  
            + 3 * second_d * first_d * (1 - inv)
            + 2 * (first_d ** 3)
        )
        
        second_half_numerator = (
            third_d * (inv ** 2)
            - 3 * second_d * first_d * inv 
            + 2 * (first_d ** 3)
        )
        
        return (   
            (x ** 3) * (1-y) * (first_half_numerator) / ((1 - inv) ** 3) - 
            x**3 * y * second_half_numerator / (inv ** 3)
        )

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
        return (inv) * (1 - inv) * (1 - 2* inv)
    
    def third_derivative_quantile(self, x, b):
        temp_exp = np.exp(x * b)
        return (temp_exp) * (temp_exp ** 2 - 4 * temp_exp + 1) / ((temp_exp + 1) ** 4)

class Probit_Link(Link_Function_For_Binary_Response):
    def quantile(self, x, b):
        return norm.cdf(x * b)
    
    def derivative_quantile(self, x, b):
        return norm.pdf(x * b)
    
    def second_derivative_quantile(self, x, b):
        eval_point = x * b
        return (-eval_point) * norm.pdf(eval_point)
    
    def third_derivative_quantile(self, x, b):
        eval_point = x * b
        return (eval_point**2 - 1) * norm.pdf(eval_point)

class CLogLogLink(Link_Function_For_Binary_Response):
    def quantile(self, x, b):
        return 1 - np.exp(-np.exp(x * b))
    
    def derivative_quantile(self, x, b):
        eval_point = x * b
        return np.exp(eval_point - np.exp(eval_point))
    
    def second_derivative_quantile(self, x, b):
        eval_point = x * b
        return (1 - np.exp(eval_point)) * np.exp(eval_point - np.exp(eval_point))
    
    def third_derivative_quantile(self, x, b):
        eval_point = x * b
        return (np.exp(2 * eval_point) - 3 * np.exp(eval_point) + 1) * np.exp(eval_point - np.exp(eval_point))
        
        