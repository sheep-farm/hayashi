# reshape — conversão entre formatos wide e long
# Essencial para preparação de dados de painel

# ══════════════════════════════════════════════════════════════════════════════
# WIDE → LONG:  reshape(df, "long", stubs=[...], i=id, j=nova_col_tempo)
# ══════════════════════════════════════════════════════════════════════════════
#
# Exemplo: dados de emprego e salário por pessoa, 3 anos (formato wide)
# Colunas: id, emp1990, emp1991, emp1992, sal1990, sal1991, sal1992
#
# load "psid_wide.csv" as wide
# let long = reshape(wide, "long",
#                    stubs=["emp", "sal"],
#                    i="id", j="year")
# print(long)
# Resultado: id, year, emp, sal  (3× mais linhas)

# ══════════════════════════════════════════════════════════════════════════════
# LONG → WIDE:  reshape(df, "wide", values=[...], i=id, j=col_tempo)
# ══════════════════════════════════════════════════════════════════════════════
#
# Reverte a operação acima
# load "psid_long.csv" as long
# let wide2 = reshape(long, "wide",
#                     values=["emp", "sal"],
#                     i="id", j="year")
# print(wide2)
# Resultado: id, emp1990, emp1991, emp1992, sal1990, sal1991, sal1992

# ── Exemplo completo com painel.csv (já em formato long) ───────────────────────
load "painel.csv" as painel

summarize(painel, lucro, alavancagem, tamanho)

# 1. Estimar modelo em formato long (natural para estimadores de painel)
let m_fe = fe(lucro ~ alavancagem + tamanho, painel, id=empresa, time=ano)
print(m_fe)

# 2. Converter para wide (útil para cross-section ou exportar)
let wide = reshape(painel, "wide",
                   values=["lucro", "alavancagem", "tamanho"],
                   i="empresa", j="ano")
print(wide)

# 3. Converter de volta para long e verificar consistência
let long2 = reshape(wide, "long",
                    stubs=["lucro", "alavancagem", "tamanho"],
                    i="empresa", j="ano")

# Deve ter as mesmas dimensões do painel original
summarize(long2, lucro, alavancagem, tamanho)

# ── Caso de uso: CPS/PSID com múltiplas ondas ──────────────────────────────────
# Muitas bases de dados de levantamentos repetem-se em ondas (waves)
# com variáveis nomeadas var1994, var1996, var1998, ...
#
# load "cps_wages.csv" as cps
# let cps_long = reshape(cps, "long",
#                         stubs=["wage", "hours", "educ"],
#                         i="id", j="year")
# let m_re = re(wage ~ hours + educ, cps_long, id=id, time=year)
# print(m_re)
