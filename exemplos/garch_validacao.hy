# Validação GARCH — retornos do S&P 500
# Dataset: sp500.dta (Stata) — retornos diários 1988-1998
# Referência: Nelson (1991) EGARCH; Glosten-Jagannathan-Runkle (1993) GJR
# Resultado esperado: GARCH(1,1) com α+β próximo de 1 (persistência de volatilidade)

load "https://www.stata-press.com/data/r14/sp500.dta" as sp

list(sp, 5)
count sp

# ── GARCH(1,1) com erros normais — especificação canônica
# Esperado: ω pequeno, α ≈ 0.05-0.15, β ≈ 0.80-0.90
let m_garch = garch(sp, change, p=1, q=1)
print(m_garch)

# forecast de volatilidade 5 dias à frente
forecast_vol(m_garch, steps=5)

# ── GARCH(1,1) com erros Student-t — captura caudas pesadas
let m_garch_t = garch(sp, change, p=1, q=1, dist=t)
print(m_garch_t)

# ── EGARCH(1,1) — assimetria (leverage effect)
# Esperado: parâmetro γ < 0 (choques negativos elevam mais a volatilidade)
let m_egarch = egarch(sp, change, p=1, q=1)
print(m_egarch)
forecast_vol(m_egarch, steps=5)

# ── GJR-GARCH(1,1) — alternativa ao EGARCH para leverage
# Esperado: parâmetro de alavancagem γ > 0
let m_gjr = gjrgarch(sp, change, p=1, q=1)
print(m_gjr)
forecast_vol(m_gjr, steps=5)
