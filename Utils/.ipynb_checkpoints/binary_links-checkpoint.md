## Link_Function_For_Binary_Response class:

Here, I briefly note the computational derivations for the observed Fisher Information, and its derivative with respect to beta.

For an individual data-point, for a binary GLM, we observe that for the quantile function $h$ (i.e., the inverse link function), the likelihood of a datapoint (given $\beta$, $x$, $y$) is simply:

$$
\begin{align*}
    L(\beta) = h(x'\beta)^{y}(1-h(x'\beta))^{1-y}.
\end{align*}
$$

This implies a log-likelihood of:

$$\ell(\beta) = y\log(h(x'\beta)) + (1-y)\log(1-h(x'\beta)).$$

Assume that $x$ is univariate for simplicity. Then, writing $h = h(x\beta), h' = h'(x\beta), \ldots$ for simplicity:

$$\ell'(\beta) = y\times x \times \frac{h'}{h} - (1-y) \times x \times \frac{h'}{1-h}.$$

Therefore:

$$\ell''(\beta) = y\times x^2 \times \frac{h''h - (h')^2}{h^2} - (1-y) \times x^2 \times \frac{h''(1-h) + (h')^2}{(1-h)^2}.$$

Therefore, the observed unit Fisher information is:

$$I(\beta) = -\ell''(\beta) = (1-y) \times x^2 \times \frac{h''(1-h) + (h')^2}{(1-h)^2} - y\times x^2 \times \frac{h''h - (h')^2}{h^2}.$$

Also, this means that:

$$
\begin{align*}
    I'(\beta) &= (1-y) \times x^3 \times \left(\frac{h'''(1-h) - h''h' + 2h'h''}{(1-h)^2} + 2h' \frac{h''(1-h) + (h')^2}{(1-h)^3}\right) \\
    &- y\times x^3 \times \left(\frac{h'''h + h''h' - 2h''h'}{h^2} - 2 h' \frac{h''h - (h')^2}{h^3}\right) \\
    &= (1-y) \times x^3 \times \left(\frac{h'''(1-h)^2 + 3h''h'(1-h) + 2(h')^3}{(1-h)^3}\right) \\
    &- y\times x^3 \times \left(\frac{h'''h^2 - 3h''h'h + 2(h')^3}{h^3}\right) 
\end{align*}
$$
