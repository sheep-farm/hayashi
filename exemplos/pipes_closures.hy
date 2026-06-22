# Pipe operator, closures, match, f-strings, collections
# Demonstrates modern language features

# ── Pipe basics ──────────────────────────────────────────────
let nums = [5, 3, 8, 1, 9, 2, 7, 4, 6]

let sorted = nums |> sort
let sorted_str = join(sorted, " ")
display f"sorted: {sorted_str}"

let top3 = nums |> sort |> reverse |> slice(0, 3)
let top3_str = join(top3, " ")
display f"top 3: {top3_str}"

let result = nums |> filter(|x| x > 4) |> sort |> map(|x| x * 10)
let result_str = join(result, " ")
display f"filtered & scaled: {result_str}"

# ── Closures ─────────────────────────────────────────────────
let square = |x| x * x
let add = |a, b| a + b

display f"square(7) = {square(7)}"
display f"add(3, 4) = {add(3, 4)}"

# closure captures outer scope
const FACTOR = 100
let scaled = [1, 2, 3] |> map(|x| x * FACTOR)
let scaled_str = join(scaled, " ")
display f"scaled by {FACTOR}: {scaled_str}"

# ── Match ────────────────────────────────────────────────────
for code in [1, 2, 3, 99] {
    let label = match code {
        1 => "low",
        2 => "medium",
        3 => "high",
        _ => "unknown"
    }
    display f"code {code} => {label}"
}

# ── If-expression ────────────────────────────────────────────
for x in [-2, 0, 5] {
    let sign = if x > 0 { "positive" } else { if x == 0 { "zero" } else { "negative" } }
    display f"{x} is {sign}"
}

# ── In operator ──────────────────────────────────────────────
let primes = [2, 3, 5, 7, 11, 13]
for n in range(1, 15) {
    let tag = if n in primes { "prime" } else { "" }
    if tag == "prime" {
        display f"  {n} is prime"
    }
}

# ── Dict ─────────────────────────────────────────────────────
let config = {"alpha": 0.05, "n_boot": 1000, "method": "HC3"}
let alpha = config["alpha"]
let nboot = config["n_boot"]
display f"significance level: {alpha}"
display f"bootstrap reps: {nboot}"

let updated = config |> dict_set("n_boot", 5000)
let nboot2 = updated["n_boot"]
display f"updated n_boot: {nboot2}"
let keys_str = join(keys(updated), " ")
display f"keys: {keys_str}"

# ── F-strings ────────────────────────────────────────────────
const PI = 3.14159265
display f"pi = {PI:.4f}"
display f"pi * 2 = {PI * 2:.2f}"
display f"scientific: {0.000123:.2e}"

# ── Type system ──────────────────────────────────────────────
let values = [42, 3.14, "text", true, [1, 2], {"a": 1}]
for v in values {
    display f"  {v} => {type(v)}"
}

# ── Try/catch ────────────────────────────────────────────────
try {
    let x = int("not_a_number")
} catch e {
    display f"caught error: {e}"
}
display "script continues after error"

# ── Pipe with data ───────────────────────────────────────────
load "https://www.stata-press.com/data/r9/auto.dta" as auto

let expensive = filter(auto, price > 10000)
summarize(expensive, price)

let m = ols(price ~ mpg + weight, auto)
print(m)
