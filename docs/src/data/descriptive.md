# Descriptive Statistics

## summarize

Basic summary (mean, sd, min, max, N):

```
summarize df wage, educ, exper
```

Detailed output adds percentiles, skewness, kurtosis:

```
summarize df wage, detail=true
```

## describe

Structure overview — column names, types, non-missing counts:

```
describe df
```

## tabulate

One-way frequency table:

```
tabulate df industry
```

Two-way cross-tabulation with chi-squared test:

```
tabulate df female, union, chi2=true
```

## correlate

Correlation matrix (pairwise complete):

```
correlate df wage, educ, exper, tenure
```

## pwcorr

Pairwise correlations with significance stars:

```
pwcorr df wage, educ, exper, tenure, stars=0.05
```

Output marks coefficients significant at the chosen level.

## ttest

One-sample test against a hypothesized mean:

```
ttest df wage, mu=50000
```

Two-sample test (equal variances assumed by default):

```
ttest df wage, by=female
ttest df wage, by=female, unequal=true
```

Paired test:

```
ttest df score_pre, score_post, paired=true
```

## ci

Confidence interval for a variable mean:

```
ci df wage
ci df wage, level=0.99
```

## centile

Report specific percentiles:

```
centile df wage, p=[10, 25, 50, 75, 90]
```

## count

Row counts, optionally filtered:

```
count df
count df if wage > 0 & !missing(educ)
```

## list

Print rows to the terminal:

```
list df, n=10
list df wage, educ if exper > 20, n=5
```
