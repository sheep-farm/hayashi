# Validação archtest — Engle (1982) ARCH LM test
# Dataset: sp500.dta — retornos diários S&P 500 1988-1998
# Esperado: rejeita H₀ (efeitos ARCH presentes em retornos financeiros)

load "https://www.stata-press.com/data/r14/sp500.dta" as sp

# ── teste na série bruta de retornos
# Lags 5 (padrão) — Stata: estat archlm, lags(5)
archtest(sp, change, lags=5)

# ── teste com mais lags
archtest(sp, change, lags=10)

# ── teste nos resíduos do GARCH(1,1)
# Se GARCH for bem especificado, H₀ não deve ser rejeitada nos resíduos padronizados
let m = garch(sp, change, p=1, q=1)
print(m)
archtest(m, lags=5)
