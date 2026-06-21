# Inferência causal: RD, Fuzzy RD, PSM, Controle Sintético
# Datasets:
#   dados.csv   — sintético cross-section (n=200)
#   painel.csv  — sintético panel (10 × 5)

# ══════════════════════════════════════════════════════════════════════════════
# 1. REGRESSÃO DESCONTÍNUA (Sharp RD)
# ══════════════════════════════════════════════════════════════════════════════
# Cenário: programa de bonificação salarial para funcionários com ≥ 10 anos de
# experiência (threshold de elegibilidade). Identificação: salários ao redor de
# xp = 10 devem ser contínuos exceto pelo salto causado pelo bônus.

load "dados.csv" as df

summarize(df, salario, experiencia, educacao, idade)

# Bandwidth automático (seletor IK — Imbens-Kalyanaraman 2012)
# Kernel triangular, polinômio local linear (padrão)
let rd_auto = rd(salario ~ experiencia, 10.0, df)
print(rd_auto)

# Bandwidth explícito, polinômio quadrático
let rd_q2 = rd(salario ~ experiencia, 10.0, df, bw=3.0, poly=2)
print(rd_q2)

# Kernel uniforme (todos os pesos iguais dentro da janela)
let rd_uni = rd(salario ~ experiencia, 10.0, df, bw=2.5, kernel="uniform")
print(rd_uni)


# ══════════════════════════════════════════════════════════════════════════════
# 2. REGRESSÃO DESCONTÍNUA FUZZY
# ══════════════════════════════════════════════════════════════════════════════
# Cenário: elegibilidade (xp ≥ 10) não garante receber o bônus (non-compliance).
# Criamos um indicador de tratamento efetivo com fuzzy assignment.
generate df recebeu_bonus = (experiencia >= 10)
# Na prática: recebeu_bonus é correlacionado com elegibilidade mas ≠ determinístico
# Estimador: LATE = salto(salario) / salto(recebeu_bonus)
let rd_fuzzy = fuzzy_rd(salario ~ experiencia, "recebeu_bonus", 10.0, df, bw=3.0)
print(rd_fuzzy)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROPENSITY SCORE MATCHING (PSM)
# ══════════════════════════════════════════════════════════════════════════════
# Cenário: efeito de alta escolaridade (educacao > 14 anos) sobre salário.
# Tratamento: educacao > 14 (pós-graduação vs graduação)
# Covariáveis de balanceamento: experiencia, idade, genero binário

generate df alta_edu = (educacao > 14)

# filter() — seleção de linhas por condição (inclui comparação de colunas string)
# Sintaxe: filter(df, condição_numérica_ou_string)
# Exemplo com coluna numérica: apenas trabalhadores adultos
let df_adultos = filter(df, idade >= 25)
# Exemplo com coluna string (se existir): filter(df, genero == "F")

# 1:1 matching sem reposição, bandwidth automático do PS
let m_psm = psm(salario ~ alta_edu + experiencia + idade, df_adultos)
print(m_psm)

# 2:1 matching com caliper 0.05 (evita matches ruins)
let m_psm2 = psm(salario ~ alta_edu + experiencia + idade, df,
                 k=2, caliper=0.05, replace=false, boot=300)
print(m_psm2)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONTROLE SINTÉTICO (Abadie-Diamond-Hainmueller 2010)
# ══════════════════════════════════════════════════════════════════════════════
# Dataset em painel longo: uma unidade tratada vs pool de controles
# Sintaxe: synth("outcome", "treated_id", t0, df, id=col, time=col)

load "painel.csv" as painel

summarize(painel, lucro, alavancagem, tamanho)

# Empresa 1 recebeu algum tratamento a partir de 2021
# t0 = 2021 → 2018-2020 são períodos pré-tratamento
let m_synth = synth("lucro", "1", 2021, painel, id="empresa", time="ano")
print(m_synth)

# Com covariáveis para melhorar o ajuste pré-tratamento (matching de preditores)
let m_synth2 = synth("lucro", "1", 2021, painel,
                     id="empresa", time="ano",
                     covs=["alavancagem", "tamanho"])
print(m_synth2)

# ── Exemplo com dados reais (California anti-tabaco — Abadie 2010) ─────────
# Dataset: Prop 99 (Proposition 99, California 1989)
# Requer dataset em formato longo com: state, year, cigsale
# load "california_prop99.csv" as ca_data
# let ca_synth = synth("cigsale", "California", 1989, ca_data,
#                      id="state", time="year",
#                      covs=["lnincome", "age15to24", "beer"])
# print(ca_synth)
