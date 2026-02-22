## compute_sum_G_i:

Here, I briefly mention the derivations of $\frac{d\mu}{dg}$ for the function compute_sum_G_i.

Note that for the probit link, the link function is $g(\mu) = \Phi^{-1}(\mu)$, and so: 
$$\frac{dg}{d\mu} = \frac{1}{\phi(\Phi^{-1}(\mu))} = \frac{1}{\phi(x'\beta)} \implies \frac{d\mu}{dg} = \phi(x'\beta).$$

Additionally, for the cloglog link, the link function is $g(\mu) = \ln(-\ln(1-\mu))$, and so: 
$$\frac{dg}{d\mu} = \frac{1}{\ln(1 - \mu)(\mu - 1)}  \implies \frac{d\mu}{dg} = \ln(1 - \mu)(\mu - 1).$$

