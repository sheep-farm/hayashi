# Functions & Closures

## Defining functions

```
fn square(x) {
    return x * x
}

display square(5)    // 25
```

Parameters are `const` by default -- the function cannot reassign them. Use `let` to create a mutable local copy:

```
fn normalize(val, mu, sigma) {
    let z = (val - mu) / sigma
    return z
}
```

## Return

`return` exits the function with a value. Without `return`, the function returns `nil`.

```
fn sign(x) {
    if x > 0 { return 1 }
    if x < 0 { return -1 }
    return 0
}
```

## Closures

Anonymous functions with `|params| body` syntax. Closures capture variables from the enclosing scope.

```
let double = |x| x * 2
display double(5)        // 10

let factor = 100
let scale = |x| x * factor
display scale(3)         // 300
```

## First-class functions

Functions and closures are values. They can be assigned to variables, stored in lists, and passed as arguments.

```
fn apply(f, x) {
    return f(x)
}

display apply(|x| x + 1, 10)    // 11
display apply(square, 4)         // 16
```

## map and filter

`map` applies a function to each element. `filter` keeps elements that satisfy a predicate. Both return new lists.

```
let nums = [1, 2, 3, 4, 5]

let doubled = map(nums, |x| x * 2)       // [2, 4, 6, 8, 10]
let evens = filter(nums, |x| x % 2 == 0) // [2, 4]
```

Named functions work too:

```
fn is_positive(x) { return x > 0 }

let pos = filter([-1, 0, 2, -3, 4], is_positive)  // [2, 4]
```

## Recursion

Functions can call themselves:

```
fn factorial(n) {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}

display factorial(10)    // 3628800
```

## Practical example

Build a list of models in a loop, then compare:

```
let specs = ["educ", "educ + exper", "educ + exper + tenure"]
let models = []

for s in specs {
    push(models, ols("lwage ~ " + s, df))
}

esttab(models)
```
