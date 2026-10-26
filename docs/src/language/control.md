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
collected into a list.

**As an expression** (preferred — result goes to a named variable):

```
let results = parallel for t in tickers {
    let sub = filter(df, ticker == t)
    nrow(sub)
}
// results is a list of nrow values, one per ticker
```

**As a statement** (result stored in the iteration variable):

```
parallel for t in tickers {
    let sub = filter(df, ticker == t)
    nrow(sub)
}
// t now holds a list of nrow values
```

Optional `threads=N` limits the number of worker threads (default: all
available CPUs):

```
let results = parallel for t in tickers, threads=4 {
    // at most 4 iterations run concurrently
    load "data.db" as sub, query=f"SELECT * FROM prices WHERE ticker = '{t}'"
    nrow(sub)
}
```

### Isolation model

Each thread gets its own interpreter with a **snapshot** of the outer
environment (subject to the capture policy below). Auto-display is
suppressed inside threads — only explicit `print()` calls produce
output. This has important consequences:

- **Reads from outer scope work**: variables defined before the
  `parallel for` are visible inside the body.
- **Writes to outer scope are silently discarded**: reassigning an outer
  variable inside the body has no effect on the outer scope. The
  mutation lives only in that thread's local environment and is lost
  when the thread exits.
- **Auto-display is suppressed**: the body's last expression is
  collected as a result, not printed. Use `print()` explicitly if you
  need output from individual iterations.

```
let n = 0
parallel for i in 1..3 {
    n = 10          // no effect on outer n
    i * 2
}
print(n)            // still 0
```

**Best practice**: treat the body as a **pure function** — read outer
variables, compute a result, and `return` it. Do not rely on side
effects to propagate outward.

### What can and cannot be captured

| Type | Capturable? | Reason |
|---|---|---|
| `Int`, `Float`, `Str`, `Bool`, `Nil` | Yes | Primitive, copied by value |
| `List`, `Dict`, `Series` (of capturable values) | Yes | `Arc`-backed, checked recursively |
| `DataFrame` | Yes | `Arc`-backed, operations are functional |
| `UserFn` | Yes | `Arc`-backed |
| Model results (`ols`, `iv`, `rolling`, ...) | No; iterator items only | Preserves the environment capture policy |

**Captured environment variables** (outer scope) are cloned into each worker
subject to the capture policy above. Dedicated models, diagnostics and error
values are not captured, even when nested in containers or generic model
metadata. **Iterator items** and returned values have no such restriction.

All shared values, including model results, use `Arc`, and model implementations
must satisfy Rust's `Send + Sync` bounds. Safety is checked by the compiler,
without unsafe thread-transfer wrappers. Repeated references to the same model
in different items are safe, including references nested in lists or dicts.
This also lets you chain two `parallel for` blocks: the first produces model
results, and the second consumes them while the originals remain usable.

```
// Capturing a model from the outer scope: excluded by capture policy
let m = ols(y ~ x, df)
parallel for i in 1..10 { predict(m, newdata) }   // m is not captured

// Iterating over model results: OK (shared allocations use Arc)
let models = parallel for t in tickers {
    let sub = filter(df, ticker == t)
    ols(y ~ x, sub)
}
let summaries = parallel for m in models {
    tidy(m)
}
```

If you need to capture a model from the outer scope, extract the data you
need (e.g. `tidy(m)`) into a capturable value first, or restructure so the
model is an iterator item rather than a captured variable.

### Skipping iterations

Use `return nil` to skip an iteration. `nil` entries are kept in the
result list (in order). `rbind()` silently skips `nil` when concatenating
DataFrames.

### Combining with rbind

The typical pattern for batch processing is `parallel for` + `rbind`:

```
let results = parallel for i in 0..n, threads=8 {
    let t = tickers[i]
    if t == "SPY" { return nil }
    let df_t = compute_betas(t)
    df_t
}
let all_betas = rbind(results)
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
