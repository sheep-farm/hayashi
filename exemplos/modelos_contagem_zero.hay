# ZIP e ZINB — Modelos de Contagem com Excesso de Zeros
# Motivação: dados de contagem (y ≥ 0 inteiro) onde P(y=0) excede o previsto por Poisson/NegBin
# Exemplos: número de visitas ao médico, patentes por empresa, acidentes de trabalho

# ══════════════════════════════════════════════════════════════════════════════
# Modelo ZIP (Zero-Inflated Poisson)
# ══════════════════════════════════════════════════════════════════════════════
# Dois componentes:
#   (1) Equação binária (Logit):   P(zero estrutural) ~ inflate vars
#   (2) Equação de contagem (Poisson): E[y | y > 0] ~ count vars
#
# Interpretação: Zero pode vir de dois processos distintos —
#   zeros "estruturais" (processo que nunca gera eventos) ou
#   zeros "amostrais"  (Poisson que acidentalmente deu zero)
#
# AIC/BIC menor que Poisson/NegBin → excesso de zero justifica o modelo

input saude
visitas renda doenca seguro idade
0.0 3.2 0.0 0.0 25.0
0.0 2.8 0.0 0.0 30.0
3.0 4.1 1.0 1.0 45.0
0.0 3.5 0.0 1.0 28.0
5.0 4.5 1.0 1.0 55.0
0.0 2.5 0.0 0.0 22.0
2.0 3.9 1.0 1.0 40.0
0.0 3.0 0.0 0.0 35.0
1.0 3.3 0.0 1.0 38.0
0.0 2.7 0.0 0.0 27.0
4.0 4.3 1.0 1.0 50.0
0.0 3.1 0.0 0.0 32.0
0.0 2.6 0.0 0.0 24.0
6.0 4.8 1.0 1.0 60.0
1.0 3.4 0.0 1.0 36.0
0.0 2.9 0.0 0.0 29.0
3.0 4.0 1.0 1.0 42.0
0.0 3.6 0.0 0.0 33.0
0.0 2.4 0.0 0.0 21.0
7.0 5.0 1.0 1.0 65.0
2.0 3.7 1.0 0.0 44.0
0.0 3.2 0.0 0.0 26.0
0.0 2.8 0.0 0.0 31.0
1.0 3.5 0.0 1.0 37.0
4.0 4.2 1.0 1.0 48.0
0.0 2.6 0.0 0.0 23.0
0.0 3.0 0.0 0.0 34.0
5.0 4.6 1.0 1.0 58.0
1.0 3.8 1.0 0.0 41.0
0.0 2.7 0.0 0.0 28.0
end

# Variáveis:
#   visitas : número de consultas médicas (contagem, muitos zeros)
#   renda   : renda familiar (log)
#   doenca  : indicador de doença crônica
#   seguro  : tem plano de saúde
#   idade   : em anos

# ── Referência: Poisson simples ───────────────────────────────────────────────
let m_pois = poisson(visitas ~ renda + doenca + seguro + idade, saude)
print(m_pois)

# ── ZIP: mesmo X para ambas as equações ───────────────────────────────────────
let m_zip = zip(visitas ~ renda + doenca + seguro + idade, saude)
print(m_zip)
# Comparar AIC: se AIC_zip < AIC_pois → excesso de zero confirmado

# ── ZIP: equação de inflação diferente ────────────────────────────────────────
# renda e seguro afetam se a pessoa "nunca vai ao médico" (zero estrutural)
# mas doenca e idade afetam só a contagem condicional
let m_zip2 = zip(visitas ~ doenca + idade, saude,
                 inflate=["renda", "seguro"])
print(m_zip2)

# ── Predições pós-ZIP ─────────────────────────────────────────────────────────
predict saude mu_zip  = m_zip, "count" # E[y|x] incondicional (inclui zeros)
predict saude pr0_zip = m_zip, "pr0" # P(y=0|x) total

summarize(saude, visitas, mu_zip, pr0_zip)

# ══════════════════════════════════════════════════════════════════════════════
# Modelo ZINB (Zero-Inflated Negative Binomial)
# ══════════════════════════════════════════════════════════════════════════════
# Como ZIP mas com NegBin na equação de contagem
# Adequado quando há overdispersion ALÉM do excesso de zeros
# Parâmetro α (dispersion) estimado junto

let m_zinb = zinb(visitas ~ renda + doenca + seguro + idade, saude)
print(m_zinb)

# ══════════════════════════════════════════════════════════════════════════════
# Escolha entre modelos
# ══════════════════════════════════════════════════════════════════════════════
# Critério: AIC/BIC impresso em cada print()
# Hierarquia típica:
#   Poisson → NegBin (se overdispersion)
#   → ZIP (se zero estrutural + Poisson)
#   → ZINB (se zero estrutural + overdispersion)

# Tabela de comparação (apenas count equation — equação de inflação via print)
# esttab(m_pois, m_zip, m_zinb)  # esttab não suporta zip/zinb (duas equações) — use print()
