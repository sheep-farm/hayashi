#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandCategory {
    Estimator,
    PostEstimation,
    Data,
    Test,
    Graph,
    Finance,
    Language,
    Collection,
    String,
}

impl CommandCategory {
    pub const fn title(self) -> &'static str {
        match self {
            CommandCategory::Estimator => "ESTIMATORS",
            CommandCategory::PostEstimation => "POST-ESTIMATION",
            CommandCategory::Data => "DATA",
            CommandCategory::Test => "TESTS",
            CommandCategory::Graph => "GRAPHS",
            CommandCategory::Finance => "FINANCE",
            CommandCategory::Language => "LANGUAGE",
            CommandCategory::Collection => "COLLECTIONS",
            CommandCategory::String => "STRINGS",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub help: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct CommandIndexGroup {
    pub category: CommandCategory,
    pub entries: &'static [&'static str],
}

pub const HELP_INDEX_GROUPS: &[CommandIndexGroup] = &[
    CommandIndexGroup {
        category: CommandCategory::Estimator,
        entries: &[
            "ols/reg  logit  probit  iv  poisson  nbreg  tobit  qreg",
            "fe  re  ab  sysgmm  pcse  xtgls  heckman  cox  km",
            "lasso  ridge  elasticnet  garch  egarch  arima  autoreg  ardl  kalman  var  vecm  svar",
            "did  rd  synth  psm  glm  rlm  gee  betareg  mixed  glsar",
            "ologit  oprobit  mlogit  cloglog  zip  zinb",
        ],
    },
    CommandIndexGroup {
        category: CommandCategory::PostEstimation,
        entries: &[
            "test  nlcom  margins  predict  esttab  estat  hausman  lincom",
            "bootstrap  bootse  influence  vif  irf  fevd  coefplot",
        ],
    },
    CommandIndexGroup {
        category: CommandCategory::Data,
        entries: &[
            "load (csv/tsv/json/dta/xlsx/parquet/sqlite/odbc)  export",
            "generate  mutate  replace  drop  keep/select  dropna  rename  sort  filter",
            "merge  append  collapse  group_by  reshape  pivot_longer  pivot_wider",
            "encode  decode  winsor  tabgen",
            "summarize  tabulate  tabstat  xtsum  ttest  correlate  pwcorr",
            "list  describe  codebook  count  ci  centile  duplicates  label  recode",
        ],
    },
    CommandIndexGroup {
        category: CommandCategory::Test,
        entries: &[
            "adf  kpss  pp  ljungbox  archtest  granger  johansen",
            "bptest  white  jb  reset  dw  swilk  sfrancia  sktest",
        ],
    },
    CommandIndexGroup {
        category: CommandCategory::Graph,
        entries: &[
            "scatter  histogram  boxplot  kdensity  acfplot  pacfplot",
            "qqplot  corrplot  graph_scatter  graph_line  graph_hist  graph_coef",
        ],
    },
    CommandIndexGroup {
        category: CommandCategory::Finance,
        entries: &["fmb  portsort  doublesort"],
    },
    CommandIndexGroup {
        category: CommandCategory::Language,
        entries: &[
            "let  const  if/else  for  while  fn  return  display  input",
            "source  export  print  quietly  capture  assert  timer  set_seed",
        ],
    },
    CommandIndexGroup {
        category: CommandCategory::Collection,
        entries: &[
            "List: push pop insert remove clear reverse index slice join",
            "      map unique flatten sort range len",
            "Dict: {\"k\": v}  keys values has_key dict_set dict_remove dict_merge",
        ],
    },
    CommandIndexGroup {
        category: CommandCategory::String,
        entries: &[
            "upper lower trim substr split str_replace contains",
            "regexm regexr regexra regexs",
        ],
    },
];

pub const COMMAND_SPECS: &[CommandSpec] = &[
    CommandSpec {
        name: "ols",
        aliases: &["reg", "regress"],
        help: concat!(
            "ols(formula, df [, options])\n",
            "  Aliases: reg, regress\n\n",
            "  Options:\n",
            "    cov=nonrobust|HC1|HC2|HC3|HC4|robust   (default: nonrobust)\n",
            "    cluster=var       Cluster-robust SEs (one-way)\n",
            "    cluster2=var      Two-way clustering\n",
            "    nw=lags           Newey-West HAC\n\n",
            "  Example:\n",
            "    let m = ols(Y ~ X1 + X2, df)\n",
            "    let m = ols(Y ~ X1 + X2, df, cluster=firm)\n",
            "    print(m)\n",
        ),
    },
    CommandSpec {
        name: "bootstrap",
        aliases: &["boot"],
        help: concat!(
            "bootstrap(estimator, formula, df, n=1000)\n",
            "  Generic bootstrap - works with any estimator.\n\n",
            "  Example:\n",
            "    bootstrap(ols, Y ~ X1 + X2, df, n=500)\n",
            "    bootstrap(logit, Y ~ X1, df, n=1000, alpha=0.10)\n",
        ),
    },
    CommandSpec {
        name: "fe",
        aliases: &[],
        help: concat!(
            "fe(formula, df [, id=col])\n",
            "  Fixed Effects (within estimator).\n",
            "  If xtset(df, id, time) declared, id= is optional.\n\n",
            "  Example:\n",
            "    xtset(df, firm, year)\n",
            "    let m = fe(Y ~ X1 + X2, df)\n",
        ),
    },
    CommandSpec {
        name: "xtset",
        aliases: &[],
        help: concat!(
            "xtset(df, id_col, time_col)\n",
            "  Declare panel structure. After xtset, panel estimators\n",
            "  (fe, re, ab, etc.) don't need id=/time=.\n\n",
            "  Example:\n",
            "    xtset(df, firm, year)\n",
            "    let m = fe(Y ~ X1 + X2, df)\n",
        ),
    },
    CommandSpec {
        name: "quietly",
        aliases: &["quiet"],
        help: concat!(
            "quietly(expr)\n",
            "  Evaluate expression without printing output.\n\n",
            "  Example:\n",
            "    quietly(ols(Y ~ X, df))\n",
            "    let m = quietly(ols(Y ~ X, df))\n",
        ),
    },
    CommandSpec {
        name: "capture",
        aliases: &["cap"],
        help: concat!(
            "capture(expr)\n",
            "  Evaluate expression ignoring errors.\n",
            "  Returns result on success, Nil on error.\n\n",
            "  Example:\n",
            "    capture(load \"maybe.csv\" as df)\n",
            "    capture(ols(Y ~ X, df))\n",
        ),
    },
    CommandSpec {
        name: "assert",
        aliases: &[],
        help: concat!(
            "assert(cond [, msg])\n",
            "  Error if condition is false.\n\n",
            "  Example:\n",
            "    assert(n > 0)\n",
            "    assert(n > 0, \"empty sample\")\n",
        ),
    },
    CommandSpec {
        name: "duplicates",
        aliases: &[],
        help: concat!(
            "duplicates(df, var [, action=report|drop|tag])\n",
            "  Report, remove, or tag duplicates.\n\n",
            "  Actions:\n",
            "    report (default) - count duplicates\n",
            "    drop  - remove duplicate rows\n",
            "    tag   - generate _dup column with count\n\n",
            "  Example:\n",
            "    duplicates(df, id)\n",
            "    duplicates(df, id, action=drop)\n",
        ),
    },
    CommandSpec {
        name: "format",
        aliases: &["fmt"],
        help: concat!(
            "format(value, fmt_str)\n",
            "  Format numeric value as string.\n\n",
            "  Example:\n",
            "    display format(3.14159, \"%.2f\")  // \"3.14\"\n",
            "    let s = format(gdp, \"%.0f\")\n",
        ),
    },
    CommandSpec {
        name: "label",
        aliases: &[],
        help: concat!(
            "label(df, var, \"description\")\n",
            "  Store label for a DataFrame variable.\n",
            "  Labels appear in describe().\n\n",
            "  Example:\n",
            "    label(df, lnY, \"Log GDP per capita\")\n",
            "    describe(df)\n",
        ),
    },
];

pub fn help_index() -> String {
    let mut out = String::from("Hayashi - Applied Econometrics Language\n\n");
    for group in HELP_INDEX_GROUPS {
        out.push_str(group.category.title());
        out.push_str(":\n");
        for entry in group.entries {
            out.push_str("  ");
            out.push_str(entry);
            out.push('\n');
        }
        out.push('\n');
    }
    out.push_str("Type help(command) for details. Ex: help(ols)\n");
    out.push_str("Type help(about) for project info.\n");
    out
}

pub fn command_spec(topic: &str) -> Option<&'static CommandSpec> {
    let topic = topic.trim();
    COMMAND_SPECS
        .iter()
        .find(|spec| spec.name == topic || spec.aliases.contains(&topic))
}

pub fn command_help(topic: &str) -> Option<&'static str> {
    command_spec(topic).map(|spec| spec.help)
}
