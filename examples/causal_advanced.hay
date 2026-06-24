# DiD, Regressão Quantílica e Sobrevivência
# Demonstra: did(), qreg(), km(), cox()

# ══════════════════════════════════════════════════════════════════════════════
# 1. DIFFERENCE-IN-DIFFERENCES (2×2 canônico)
# ══════════════════════════════════════════════════════════════════════════════
# Dataset: dados de painel sintético
# Cenário: uma política pública atingiu o grupo tratado a partir de t=2
#   inlf   — resultado (outcome)
#   treated — indica grupo tratado (invariante no tempo)
#   post   — indica período pós-intervenção
#
# Identifica ATT via regressão:
#   y = β₀ + β₁*treated + β₂*post + δ*(treated × post) + ε
# δ = ATT (diferença das diferenças)

load "painel.csv" as df

# gerar indicadores se não existirem no dataset
generate df post = (ano >= 2021)
# treated: unidade 1 é tratada; demais são controle
generate df treated = (empresa == 1)

let m_did = did(lucro ~ treated + post, df)
print(m_did)

# Com erros padrão HC1 (robust)
let m_did_hc1 = did(lucro ~ treated + post, df, cov=HC1)
print(m_did_hc1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. REGRESSÃO QUANTÍLICA (Koenker-Bassett 1978)
# ══════════════════════════════════════════════════════════════════════════════
# Estima E[y | X, τ] = X'β(τ) para diferentes quantis da distribuição
# Erros padrão via bootstrap (pares)
# Útil quando o efeito varia ao longo da distribuição de y

load "dados.csv" as dados

# Mediana (τ = 0.50) — mais robusta a outliers que OLS
let m_q50 = qreg(salario ~ educacao + experiencia + idade, dados, tau=0.5)
print(m_q50)

# Cauda inferior (τ = 0.25)
let m_q25 = qreg(salario ~ educacao + experiencia + idade, dados, tau=0.25)
print(m_q25)

# Cauda superior (τ = 0.75)
let m_q75 = qreg(salario ~ educacao + experiencia + idade, dados, tau=0.75)
print(m_q75)

# Comparação OLS vs quantis (para cada regressão com 200 boots)
let m_ols = ols(salario ~ educacao + experiencia + idade, dados)
esttab(m_ols, m_q25, m_q50, m_q75)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANÁLISE DE SOBREVIVÊNCIA
# ══════════════════════════════════════════════════════════════════════════════
# Variáveis necessárias no dataset:
#   time  — tempo de sobrevivência / censura
#   event — indicador de evento (1 = ocorreu, 0 = censurado)
#   covariáveis para Cox

# ── Kaplan-Meier ──────────────────────────────────────────────────────────────
# Estimador não paramétrico de S(t) = P(T > t)
# Lida com censura à direita; mostra medianas e IC 95% por Greenwood

# km(coluna_tempo, coluna_evento, df)
# load "sobrevivencia.csv" as surv
# let m_km = km(time, event, surv)
# print(m_km)

# ── Cox Proportional Hazards ──────────────────────────────────────────────────
# Modelo semiparamétrico: h(t|X) = h₀(t) · exp(X'β)
# β interpreta-se como log-hazard ratio
# Hipótese fundamental: razão de hazard entre grupos é constante no tempo

# Sintaxe: cox(tempo ~ x1 + x2, df, event=coluna_evento)
# load "sobrevivencia.csv" as surv
# let m_cox = cox(time ~ age + treatment + stage, surv, event=event)
# print(m_cox)

# Exemplo com dados reais (cancer dataset do R/survival):
# load "https://vincentarelbundock.github.io/Rdatasets/csv/survival/cancer.csv" as cancer
# let m_km_cancer  = km(time, status, cancer)
# let m_cox_cancer = cox(time ~ age + sex + ph.ecog, cancer, event=status)
# print(m_km_cancer)
# print(m_cox_cancer)
