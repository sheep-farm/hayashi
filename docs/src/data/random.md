# Random Variates

Hayashi provides random variate generators that can be used directly in
`generate` expressions. All generators respect `set_seed` and produce one
value per row of the target DataFrame.

## Basic usage

```hay
set_seed(42)
let df = dataframe(1000)

generate df u = rnormal()
generate df eps = rnormal(0, 2)
generate df treat = rbernoulli(0.5)
```

## Continuous distributions

| Function | Parameters | Description |
|---|---|---|
| `rnormal()` or `rnormal(μ, σ)` | mean, standard deviation | Normal |
| `rlognormal(μ, σ)` | log-mean, log-standard deviation | Log-normal |
| `rskewnormal(loc, scale, shape)` | location, scale, shape | Skew-normal |
| `rcauchy(loc, scale)` | location, scale | Cauchy |
| `rstudentt(df)` or `rt(df)` | degrees of freedom | Student's t |
| `rchisq(k)` | degrees of freedom | Chi-squared |
| `rf(d1, d2)` | numerator df, denominator df | F distribution |
| `rbeta(α, β)` | two positive shape parameters | Beta |
| `rgamma(shape, scale)` | shape, scale | Gamma |
| `rexponential(λ)` | rate λ | Exponential |
| `rweibull(scale, shape)` | scale, shape | Weibull |
| `rpareto(scale, shape)` | scale (x_m), shape (α) | Pareto |
| `rpert(min, max, mode)` or `rpert(min, max, mode, shape)` | min, max, mode, optional shape | PERT |
| `rtriangular(min, max, mode)` | min, max, mode | Triangular |
| `rfrechet(location, scale, shape)` | location, scale, shape | Fréchet |
| `rgumbel(location, scale)` | location, scale | Gumbel |
| `rinversegaussian(mean, shape)` | mean, shape | Inverse Gaussian |
| `rnig(alpha, beta)` | tail heaviness, asymmetry | Normal-inverse-Gaussian |
| `runiform()` or `runiform(a, b)` | low, high | Uniform `[0,1)` or `[a,b)` |

## Discrete distributions

| Function | Parameters | Description |
|---|---|---|
| `rbernoulli(p)` | probability (default 0.5) | Bernoulli (0/1) |
| `rbinomial(n, p)` | trials, success probability | Binomial |
| `rpoisson(λ)` | mean λ | Poisson |
| `rgeometric(p)` | success probability | Geometric (failures before first success) |
| `rhypergeometric(N, K, n)` | population, successes, draws | Hypergeometric |
| `rzeta(a)` | exponent `a > 1` | Zeta |
| `rzipf(N, s)` | population size, exponent | Zipf |

## Notes

- All arguments are scalar. The generator evaluates them once and creates a
  column with the same number of rows as the DataFrame.
- Invalid parameters raise a runtime error (e.g., negative shape, probability
  outside `[0,1]`, `a <= 1` for `rzeta`).
- `rgeometric(p)` follows the `rand_distr` convention and returns the number of
  failures before the first success, with support `{0, 1, 2, ...}`.
