# The REPL

Start an interactive session by running `hay` with no arguments:

```bash
hay
```

```
Hayashi 0.x.y  — Applied Econometrics Language
In honor of Fumio Hayashi. Type 'exit' or Ctrl-D to quit.

hay>
```

## Features

**Multi-line input.** Open braces automatically continue to the next line. The prompt changes to `>` until braces are balanced:

```
hay> for c in ["nonrobust", "HC1", "HC3"] {
      >     print(c)
      > }
```

**Tab completion.** Press Tab to complete keywords, function names, and variable names currently in scope.

**Syntax highlighting.** Keywords are blue, strings green, numbers yellow, and comments gray.

**History hints.** Previous commands appear as gray ghost text (fish-style). Press the right arrow to accept.

**Persistent history.** Command history is saved to `.hay_history` in the working directory and restored on next launch.

## Example session

```
hay> input df
      > wage educ exper
      > 12.5 12 10
      > 15.2 14  8
      > 10.1 10 15
      > 18.7 16 12
      > 11.3 12  5
      > end
hay> generate df lwage = log(wage)
hay> let m = ols(lwage ~ educ + exper, df)
hay> print(m)
hay> test(m, "white")
```

Input blocks start with `input <name>`, followed by a header row, data rows, and `end`. The REPL collects all lines until `end` before executing.

## Exiting

Type `exit`, `quit`, or press Ctrl-D. Ctrl-C cancels the current multi-line input without quitting.
