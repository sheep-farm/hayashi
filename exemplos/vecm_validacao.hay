# Validação VECM — dataset Lütkepohl (macro alemão 1960q1-1982q4)
# Referência: Lütkepohl (2005) cap. 6 — VECM com rank=1
# Variáveis em nível (não diferenciadas): ln_inv, ln_inc, ln_consump

load "https://www.stata-press.com/data/r14/lutkepohl2.dta" as lut

tsset lut qtr

# ── VECM(rank=1, lags=2) — especificação de referência do livro
# beta: vetor de cointegração (relação de longo prazo)
# alpha: velocidades de ajuste (error correction)
# Esperado: autovalor dominante ≈ 0.2-0.4 para este dataset
let m = vecm(lut, ln_inv, ln_inc, ln_consump, lags=2, rank=1)
print(m)

# ── VECM(rank=2) — testa duas relações de cointegração
let m2 = vecm(lut, ln_inv, ln_inc, ln_consump, lags=2, rank=2)
print(m2)
