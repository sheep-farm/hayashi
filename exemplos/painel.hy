# Dataset: painel.csv — dados sintéticos (10 empresas × 5 anos)
# Variáveis: empresa, ano, lucro, alavancagem, tamanho, setor

load "painel.csv" as df

summarize(df, lucro, alavancagem, tamanho)

# OLS pooled (ignora estrutura de painel)
let m_ols = ols(lucro ~ alavancagem + tamanho + C(setor), df, cov=HC3)
print(m_ols)

test(m_ols, "bp")
test(m_ols, "dw")

# Fixed Effects (within transformation — elimina heterogeneidade não observada)
let m_fe = fe(lucro ~ alavancagem + tamanho, df, id=empresa)
print(m_fe)

# Random Effects (Swamy-Arora GLS)
let m_re = re(lucro ~ alavancagem + tamanho, df, id=empresa)
print(m_re)

predict df lucro_hat = m_ols
predict df residuo   = m_ols, "residuals"

export(m_ols, "latex", "/tmp/painel_ols.tex")
export(m_fe,  "txt",   "/tmp/painel_fe.txt")
