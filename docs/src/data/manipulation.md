# Data Manipulation

## Creating variables

```
generate df lwage = log(wage)
generate df exper2 = exper^2
generate df wage_real = wage / cpi * 100
generate df female = if(sex == "F", 1, 0)
```

Pipe with closures for multi-step transforms:

```
generate df z_wage = wage |> |x| (x - mean(wage)) / sd(wage)
```

## Replacing values

```
replace df wage = 0 if wage < 0
replace df industry = "Other" if n_firms < 10
```

## Filtering rows

```
filter df age >= 18 & age <= 65
filter df !missing(wage)
```

## Selecting, dropping, renaming columns

```
keep df wage, educ, exper
drop df id, timestamp
rename df earnings = wage
```

## Sorting

```
sort df year, firm_id
sort df -wage              // descending
```

## Merging and appending

```
let merged = merge(firms, returns, on="permno")
let panel = merge(firms, macro, on=["year", "country"], how="left")
let full = append(df2019, df2020)
```

`how=` accepts `"inner"` (default), `"left"`, `"right"`, `"outer"`.

### rbind — concatenate many DataFrames

`rbind` takes a list of DataFrames and stacks them vertically in one pass.
`nil` entries are silently skipped — useful when combining results from
`parallel for` where some iterations return `nil`:

```
parallel for i in 0..n, threads=8 {
    let t = tickers[i]
    if t == "SPY" { return nil }
    let df_t = compute(t)
    df_t
}
let results = rbind(i)   // i holds the list; concat all, skip nils
```

## Collapsing (group aggregation)

```
let means = collapse(df, mean(wage) mean(hours), by=industry)
let stats = collapse(df, mean(ret) sd(ret) count(ret), by=[sector, year])
```

## Reshaping

```
let long = reshape(df, "long", stub="inc", i="id", j="year")
let wide = reshape(df, "wide", stub="inc", i="id", j="year")
```

## Winsorizing

```
winsor df wage, p=0.01            // 1st and 99th percentile
winsor df ret, p=0.05, replace    // overwrite original column
```

## Encoding and dummy generation

```
encode df industry                // string -> numeric with value labels
tabgen df region                  // creates dummy columns: region_N, region_S, ...
```

## Recoding

```
recode df educ (0/8 = 1 "Primary") (9/12 = 2 "Secondary") (13/max = 3 "Tertiary")
```

## Duplicates and labels

```
duplicates df firm_id year, report
duplicates df firm_id year, drop
label df wage "Monthly wage (BRL)"
label df educ "Years of schooling"
```

## Preserve / Restore

Snapshot the DataFrame before destructive operations:

```
preserve df
filter df year == 2020
let m = ols(lwage ~ educ + exper, df)
restore df
```

## Dropping missing values

```
dropna df                         // all columns
dropna df wage, educ              // specific columns
```
