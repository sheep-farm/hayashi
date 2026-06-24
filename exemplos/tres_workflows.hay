// Três workflows — Cross-Section, Painel, Logit
// Demonstra integração das features da linguagem

// ═══════════════════════════════════════════════════
// 1. CROSS-SECTION: OLS com diagnósticos
// ═══════════════════════════════════════════════════
input cs
wage educ exper tenure
12.5 12 10 5
15.2 14 8 3
10.1 10 15 7
18.7 16 12 6
11.3 12 5 2
20.5 18 20 10
13.8 14 6 4
9.5 10 3 1
16.4 16 10 8
14.2 14 12 5
17.1 16 15 9
11.8 12 7 3
end

let m1 = reg(wage ~ educ + exper + tenure, cs)
let m2 = reg(wage ~ educ + exper, cs)
esttab(m1, m2)
test(m1, "exper", "tenure")
coefplot(m1)

// ═══════════════════════════════════════════════════
// 2. PAINEL: FE vs RE com Hausman
// ═══════════════════════════════════════════════════
input panel
output capital labor firm year
10.2 5 8 1 2019
11.0 5 9 1 2020
12.5 6 9 1 2021
11.8 5 10 1 2022
19.3 10 12 2 2019
20.1 10 13 2 2020
23.1 12 14 2 2021
20.7 11 13 2 2022
14.6 7 10 3 2019
15.3 7 11 3 2020
17.9 8 11 3 2021
15.2 7 12 3 2022
24.8 13 15 4 2019
25.5 13 16 4 2020
27.3 14 16 4 2021
26.1 14 17 4 2022
end

xtset(panel, firm, year)
let m_fe = fe(output ~ capital + labor, panel)
let m_re = re(output ~ capital + labor, panel)
esttab(m_fe, m_re)
hausman(m_fe, m_re)

// ═══════════════════════════════════════════════════
// 3. LOGIT: Modelo binário com efeitos marginais
// ═══════════════════════════════════════════════════
// Logit requer dados sem separação perfeita.
// Com amostras pequenas inline, separação é quase inevitável.
// Para demonstrar: usar dados reais via load.
// Exemplo com Cattaneo:
//   load "cattaneo2.dta" as bin
//   generate bin low_bw = (bweight < 2500)
//   let m_logit = logit(low_bw ~ mbsmoke + mage + medu, bin)
//   margins(m_logit)
//   coefplot(m_logit)
