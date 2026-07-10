#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SOURCE_BRANCH="$(git branch --show-current)"
SOURCE_DIR="site"
TARGET_BRANCH="gh-pages"
DOMAIN="haylang.dev"

if [[ "$SOURCE_BRANCH" == "$TARGET_BRANCH" ]]; then
  echo "Você está no branch '$TARGET_BRANCH'. Saia dele antes de rodar este script." >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Pasta '$SOURCE_DIR' não encontrada em $ROOT" >&2
  exit 1
fi

PUSH=0
if [[ "${1:-}" == "--push" ]]; then
  PUSH=1
fi

echo "Publicando site a partir de '$SOURCE_BRANCH'..."

# Copia o site para um diretório temporário porque o branch gh-pages não contém a pasta site/
TMP_DIR="$(mktemp -d)"
cp -a "$SOURCE_DIR"/ "$TMP_DIR/"

# Salva o branch atual e muda para gh-pages
git checkout "$TARGET_BRANCH"

# Remove tudo na raiz exceto .git e CNAME
find . -mindepth 1 -maxdepth 1 ! -name '.git' ! -name 'CNAME' -exec rm -rf {} +

# Copia o conteúdo do site do temporário para a raiz do gh-pages
cp -a "$TMP_DIR"/ .
rm -rf "$TMP_DIR"

# Garante que o CNAME existe
if [[ ! -f CNAME ]]; then
  echo "$DOMAIN" > CNAME
  echo "CNAME criado com '$DOMAIN'"
fi

# Commita as mudanças no gh-pages
if git add -A && git commit -m "site: sync from $SOURCE_BRANCH"; then
  echo "Commit de publicação criado no branch '$TARGET_BRANCH'."
else
  echo "Nada de novo para publicar."
fi

if [[ "$PUSH" -eq 1 ]]; then
  git push origin "$TARGET_BRANCH"
  echo "Site publicado em https://$DOMAIN"
else
  echo "Pronto para push. Rode novamente com '--push' para enviar ao GitHub."
fi

# Retorna ao branch de origem
git checkout "$SOURCE_BRANCH"
