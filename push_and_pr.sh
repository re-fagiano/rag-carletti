#!/bin/bash

echo "🔁 Aggiunta file modificati..."
git add .

echo "✍️ Inserimento commit..."
git commit -m "Modifiche automatiche da Codex"

echo "🚀 Push sul branch corrente..."
git push origin HEAD

echo "📬 Creazione Pull Request..."
gh pr create --fill
