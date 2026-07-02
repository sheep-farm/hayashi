# Variables & Types

## Types

Hayashi has ten types:

| Type | Literal | Example |
|---|---|---|
| `int` | integer | `42`, `-7` |
| `float` | decimal | `3.14`, `1e-5` |
| `bool` | boolean | `true`, `false` |
| `str` | string | `"hello"` |
| `nil` | absent value | `nil` |
| `list` | ordered collection | `[1, 2, 3]` |
| `dict` | key-value map | `{"a": 1, "b": 2}` |
| `dataframe` | tabular data | returned by `load` |
| `function` | user-defined fn | `fn f(x) { ... }` |
| `closure` | anonymous fn | `\|x\| x * 2` |

Introspect with `type()`:

```
display type(42)         // "int"
display type(3.14)       // "float"
display type([1, 2])     // "list"
display type({"a": 1})   // "dict"
```

## Declaration

```
let x = 10          // mutable
const PI = 3.14159  // immutable -- cannot be reassigned
```

`let` creates a mutable binding. `const` creates an immutable one. Attempting to reassign a `const` is a compile error.

## Scoping

Variables are block-scoped. A `let` inside `{ }` is not visible outside.

```
let x = 1
if true {
    let y = 2      // y exists only here
    display x + y  // 3
}
// display y      // error: undefined variable 'y'
```

**No shadowing.** Redeclaring a variable that is `const` in an outer scope is an error. This prevents subtle bugs common in Stata's global namespace.

## Function parameters

Function parameters are `const` by default -- the function cannot modify its inputs:

```
fn process(data) {
    // data = 0   // error: parameter is immutable
    let local = data * 2
    return local
}
```

## Type conversions

```
int(3.9)       // 3     (truncates)
int("42")      // 42
float(42)      // 42.0
float("3.14")  // 3.14
str(3.14)      // "3.14"
str(true)      // "true"
bool(0)        // false
bool(42)       // true
bool(nil)      // false
```

## Type predicates

```
is_int(42)        // true
is_float(3.14)    // true
is_bool(true)     // true
is_str("hello")   // true
is_list([1, 2])   // true
is_dict({"a": 1}) // true
is_df(df)         // true
is_fn(|x| x)      // true
```

## Build a DataFrame from a dict

```
let df = dataframe({"x": [1, 2, 3], "y": [4, 5, 6]})
```

## List operations

Lists are ordered, heterogeneous collections. `push` and `pop` mutate in-place; all others return a new list.

```
let nums = [3, 1, 2]
push(nums, 4)                    // nums is now [3, 1, 2, 4]
pop(nums)                        // nums is now [3, 1, 2]
let s = sort(nums)               // [1, 2, 3]
let m = map(nums, |x| x * 10)   // [30, 10, 20]
let u = unique([1, 2, 2, 3])     // [1, 2, 3]
let r = range(0, 5)              // [0, 1, 2, 3, 4]
display len(nums)                // 3
```

All 16: `push`, `pop`, `insert`, `remove`, `clear`, `reverse`, `index`, `slice`, `join`, `map`, `filter`, `unique`, `flatten`, `sort`, `range`, `len`.

## Dict operations

```
let d = {"name": "Alice", "age": 30}
display d["name"]                     // Alice
let k = keys(d)                       // ["age", "name"] (sorted)
display has_key(d, "name")            // true
let d2 = dict_set(d, "city", "SP")    // new dict with added key
let d3 = dict_merge(d, {"age": 31})   // d2 values overwrite d
```

All 7: `keys`, `values`, `has_key`, `dict_set`, `dict_remove`, `dict_merge`, `len`.
