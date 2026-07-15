# Control Flow

## if / else

Statement form:

```
if x > 0 {
    display "positive"
} else if x == 0 {
    display "zero"
} else {
    display "negative"
}
```

Expression form (returns a value, requires `else`):

```
let label = if x > 0 { "positive" } else { "negative" }
```

## match

Pattern matching expression. `_` is the wildcard (matches anything).

```
let label = match status {
    1 => "active",
    2 => "inactive",
    _ => "unknown"
}
```

Arms are checked top to bottom. The first match wins.

## for

Range loop (exclusive upper bound, like Rust/Python):

```
for i in 1..10 {
    display i    // 1, 2, ..., 9
}
```

List loop:

```
for v in ["X1", "X2", "X3"] {
    eststo(ols("Y ~ " + v, df))
}
```

## parallel for

Concurrent variant of `for`. Iterations run in parallel across threads;
each iteration's return value (last expression or explicit `return`) is
collected into a list, which is stored back into the **iteration variable**
after all threads complete.

```
parallel for t in tickers {
    let sub = filter(df, ticker == t)
    nrow(sub)
}
// t now holds a list of nrow values, one per ticker
```

Optional `threads=N` limits the number of worker threads (default: all
available CPUs):

```
parallel for t in tickers, threads=4 {
    // at most 4 iterations run concurrently
    load "data.db" as sub, query=f"SELECT * FROM prices WHERE ticker = '{t}'"
    nrow(sub)
}
```

Each thread gets its own interpreter with a snapshot of the outer
environment (only send-safe values are captured). Use `return nil` to
skip an iteration; `nil` entries are kept in the result list.

Combine with `rbind()` to aggregate per-iteration DataFrames into one:

```
parallel for i in 0..n, threads=8 {
    let t = tickers[i]
    let df_t = compute_betas(t)
    df_t
}
let all_betas = rbind(i)   // i holds the list of DataFrames
```

## while

```
let i = 0
while i < 10 {
    display i
    i = i + 1
}
```

## break / continue

`break` exits the innermost loop. `continue` skips to the next iteration.

```
for i in 1..100 {
    if i == 5 { break }
    if i % 2 == 0 { continue }
    display i    // 1, 3
}
```

## try / catch

Structured error handling. Unlike Stata's `capture`, gives access to the error message.

```
try {
    load "data.csv" as df
} catch e {
    display f"Error: {e}"
}
```

## return

Exit a function early with a value:

```
fn abs(x) {
    if x < 0 { return -x }
    return x
}
```

## Related: `quietly(expr)`, `capture(expr)`, `assert(cond, "msg")`
