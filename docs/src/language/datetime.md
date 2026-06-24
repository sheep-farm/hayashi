# Date & Time

## Parsing

Create date/time values from strings:

```
let t = date("2024-06-15")
let t = datetime("2024-06-15 14:30:00")
```

Both return a Unix timestamp (float), compatible with arithmetic and comparisons.

## Extracting components

Use inside `generate` to create columns from DateTime or timestamp columns:

```
generate df Y = year(date_col)
generate df M = month(date_col)
generate df D = day(date_col)
generate df H = hour(date_col)
generate df Min = minute(date_col)
generate df S = second(date_col)
generate df W = dow(date_col)         // 0=Monday, 6=Sunday
```

These functions work with:
- DateTime columns (parsed from CSV/Excel/DTA date fields)
- Float timestamps (Unix epoch seconds)

## Filtering by date

Use `date()` to create scalar values for comparison:

```
let cutoff = date("2020-01-01")
let sub = filter(df, date_col >= cutoff)
```

Or directly in estimation:

```
let m = ols(Y ~ X1 + X2, df, if=year(date_col) >= 2020)
```

## Practical example

Monthly panel with date extraction:

```
load "returns.csv" as df

generate df yr = year(date)
generate df mo = month(date)
generate df weekday = dow(date)

// Restrict to weekdays only
let trading = filter(df, weekday <= 4)

// Estimate by subperiod
let pre = filter(df, date < date("2020-03-01"))
let post = filter(df, date >= date("2020-03-01"))

let m_pre = ols(ret ~ mkt + smb + hml, pre)
let m_post = ols(ret ~ mkt + smb + hml, post)
esttab(m_pre, m_post)
```
