# Host API (data plane)

`hay-server` is the host-facing data plane for embedding Hayashi in a
graph/node runtime. It complements the DAP adapter, which remains the
debug/control plane.

## Build and run

```bash
cargo build --bin hay-server
./target/debug/hay-server
```

One JSON request per line on stdin, one JSON response per line on stdout.
Script output is captured and returned, so protocol stdout is never polluted.

## Sessions

A session owns a persistent `Interpreter`; variables survive across `run`
calls and can be shared by reference (important for DataFrames/models).

Commands:

| cmd | fields | result |
|---|---|---|
| `create` | — | `{"session":"s1"}` |
| `ping` | — | version |
| `set` | `session`, `name`, `value` | binds a JSON value |
| `set_ref` | `session`, `name`, `from` | aliases an existing value (no copy) |
| `get` | `session`, `name` | full JSON value |
| `get_summary` | `session`, `name` | `{kind, summary}` |
| `list` | `session` | names, kinds and summaries |
| `run` / `eval` | `session`, `source` | `{output, stderr, error}` |
| `remove` | `session`, `name` | removes one binding |
| `clear` | `session` | removes all bindings |
| `reset` | `session` | fresh interpreter (plugins kept) |
| `dispose` | `session` | drops the session |

## Value envelope

Scalars:

```json
{"kind":"number","value":10.5}
{"kind":"int","value":3}
{"kind":"bool","value":true}
{"kind":"text","value":"hello"}
{"kind":"nil"}
{"kind":"list","items":[...]}
{"kind":"dict","items":{"name":{...}}}
```

DataFrame (numeric and text columns; `null` means missing/NaN):

```json
{
  "kind":"dataframe",
  "n":3,
  "columns":[
    {"name":"x","type":"float","values":[1.0,2.0,3.0]},
    {"name":"y","type":"float","values":[3.0,5.0,7.0]}
  ]
}
```

Column types accepted on input: `float`, `int`, `bool`, `string`,
`categorical`. DateTime columns are exported as ISO text.

Model result (OLS and other estimators with a `ModelView`):

```json
{
  "kind":"model",
  "modelType":"OlsResult",
  "summary":"OLS(k=2, n=3), R2=1.0000",
  "terms":["_cons","x"],
  "estimate":[...],
  "stdError":[...],
  "statistic":[...],
  "pValue":[...],
  "confLow":[...],
  "confHigh":[...],
  "stats":{"r2":{"kind":"number","value":1.0}},
  "residuals":[...],
  "fitted":[...]
}
```

Models are not reconstructed from JSON. Inside a session, pass them with
`set_ref`; export them with `get` only for inspection.

## One-shot execution

`execute` binds inputs, runs `source`, and returns the requested outputs in a
single round trip. Each input is either a serialized `value` or a `ref` to an
existing session variable (required for models):

```json
{
  "id": 5,
  "cmd": "execute",
  "session": "s1",
  "source": "let m = ols(y ~ x, df)\n__n3_out_1 = m",
  "inputs": {
    "df": {"value": {"kind":"dataframe","n":3,"columns":[...]}}
  },
  "outputs": ["__n3_out_1"]
}
```

The result carries `output`, `stderr`, `error` and `outputs` (a map from the
requested name to its value envelope). Requested outputs are removed from the
session before the run, so a script that no longer assigns a name cannot leak
a previous value. A model input cannot be sent by `value`; use `ref`.

## Example

```
{"id":1,"cmd":"create"}
{"id":2,"cmd":"set","session":"s1","name":"df","value":{"kind":"dataframe","n":3,"columns":[{"name":"x","type":"float","values":[1,2,3]},{"name":"y","type":"float","values":[3,5,7]}]}}
{"id":3,"cmd":"execute","session":"s1","source":"let m = ols(y ~ x, df)\n__out = m","inputs":{"df":{"ref":"df"}},"outputs":["__out"]}
{"id":4,"cmd":"get","session":"s1","name":"__out"}
```

## WebAssembly

`wasm-bindgen` exposes the same session surface:

```js
const s = new WasmSession();
s.set_input_json("df", dataframeJson);
s.run("let m = ols(y ~ x, df)");
const model = JSON.parse(s.get_output_json("m"));
s.set_input_ref("alias", "m");
s.reset();
```

Output during `WasmSession.run` is captured by the Rust side; the JS print
callback is only used by the legacy `run_hayashi` entry point.

## Current limits

- JSON is the transport for now; an Arrow IPC path can be added without
  changing the command model.
- No cancellation/timeout yet; one request at a time per server.
- `ModelResult`/OLS are exported typed; other estimator variants may fall back
  to `opaque` until they implement `ModelView`.
- Native output capture uses `gag`; wasm capture uses the crate's print sink.
