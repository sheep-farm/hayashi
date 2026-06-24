# Pipe Operator

## Basics

The pipe `|>` passes the left-hand side as the first argument to the right-hand side.

```
[3, 1, 2] |> sort              // sort([3, 1, 2]) -> [1, 2, 3]
[3, 1, 2] |> sort |> reverse   // [3, 2, 1]
```

## With closures

Inline closures let you transform values mid-chain:

```
[1, 2, 3] |> map(|x| x * 10)              // [10, 20, 30]
[1, 2, 3, 4] |> filter(|x| x > 2)         // [3, 4]
[1, 2, 3] |> map(|x| x * 2) |> sort       // [2, 4, 6]
```

Pipe with a single-argument closure:

```
generate df X2 = exper |> |x| x * x
```

This is equivalent to `generate df X2 = exper^2` but demonstrates how pipes compose arbitrary transforms.

## Chaining

Pipes shine when chaining multiple operations:

```
let result = raw_data
    |> filter(|row| row > 0)
    |> sort
    |> unique
    |> map(|x| x * 100)
```

## Inside generate

Pipes work inside `generate` for column-level transforms:

```
generate df z_wage = wage |> |x| (x - mean(wage)) / sd(wage)
```

## F-strings

String interpolation with `f"..."`. Supports format specifiers.

```
let mu = 3.14159
display f"mean = {mu:.2f}"          // mean = 3.14
display f"sum = {2 + 3}"            // sum = 5
display f"{{literal braces}}"       // {literal braces}
```

Format specifiers: `{x:.Nf}` for N decimal places, `{x:.Ne}` for scientific notation.

## in operator

Membership test for lists, dicts (key check), and strings (substring).

```
display 2 in [1, 2, 3]              // true
display "name" in {"name": "A"}     // true
display "lo" in "hello"             // true

if x in [1, 2, 3] {
    display "found"
}
```

## Regex

Four regex functions for string matching and extraction:

```
display regexm("abc123", "[0-9]+")           // true (match test)
display regexr("abc123", "[0-9]+", "NUM")    // abcNUM (replace first)
display regexra("a1b2c3", "[0-9]", "X")      // aXbXcX (replace all)
display regexs("price: 42.5", "([0-9.]+)")  // 42.5 (extract group)
```

Regex works in `generate` and estimation `if=` conditions:

```
generate df is_dr = regexm(name, "^Dr")
let m = ols(Y ~ X, df, if=regexm(name, "Dr"))
```
