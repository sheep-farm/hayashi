// Regex — paridade com Stata regexm/regexr/regexs
let s = "preço: R$42.50 por unidade"

// regexm: match?
display regexm(s, "R\\$\\d+")

// regexs: extrair
display regexs(s, "R\\$(\\d+\\.\\d+)")

// regexr: substituir primeira
display regexr(s, "\\d+\\.\\d+", "99.99")

// regexra: substituir todas
let s2 = "aaa bbb aaa ccc aaa"
display regexra(s2, "aaa", "XXX")

// Uso em if
if regexm("hello@email.com", "@") {
    display "é email"
}
