# Validação VAR — dataset Lütkepohl (macro alemão trimestral 1960q1-1982q4)
# Referência: Lütkepohl (2005) "New Introduction to Multiple Time Series Analysis"
# Variáveis: ln_inv (investimento), ln_inc (renda), ln_consump (consumo)

load "https://www.stata-press.com/data/r14/lutkepohl2.dta" as lut

list(lut, 5)
count lut

# declara série temporal
tsset lut qtr

# ── VAR(2) — especificação de referência do Lütkepohl
# Esperado: AIC próximo ao do livro, coefs plausíveis
let m = var(lut, ln_inv, ln_inc, ln_consump, lags=2)
print(m)

# ── IRF — 10 trimestres
# Esperado: choque em ln_inc eleva ln_consump (efeito renda positivo)
irf(m, steps=10)

# ── FEVD — decomposição da variância
# Esperado: ln_consump explicado principalmente pelo próprio passado e ln_inc
fevd(m, steps=10)

# ── VAR(1) para comparar AIC
let m1 = var(lut, ln_inv, ln_inc, ln_consump, lags=1)
print(m1)
