# 1. Kill locking processes
taskkill /F /IM "git.exe" /T 2>$null
taskkill /F /IM "Code.exe" /T 2>$null

# 2. Remove the 3GB bloat
Write-Host "Wiping local bloat (Git and Venv)..." -ForegroundColor Cyan
Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

# 3. Start fresh
git init

# 4. Apply the .gitignore
$ignoreContent = @"
# Secrets — never commit
.env
deploy/azure.env
env.txt
session.json
.mcp.json

# Python
__pycache__/
*.py[cod]
*.pyo
*.pyc
*.egg-info/
dist/
build/
.venv/
venv/

# Model artifacts (large binaries — use Azure Blob instead)
models/
*.pkl
*.joblib
*.parquet

# Data directory (runtime generated)
/data/

# Deploy logs (runtime generated)
deploy/logs/

# Dev scripts (not needed in production)
scripts/dev/
create_test_structure.py
trader-ai-logs.txt

# Local fallback storage
/tmp/

# IDE and tool config
.vscode/
.idea/
.claude/

# OS
.DS_Store
Thumbs.db

# Test artifacts
.coverage
htmlcov/
.pytest_cache/
"@
Set-Content -Path .gitignore -Value $ignoreContent -Encoding utf8

# 5. Commit and force push
git add .
git commit -m "Fresh start: Removed 3GB bloat and optimized root-level ignoring"

# Add remote only if it doesn't already exist
if (-not (git remote get-url origin 2>$null)) {
    git remote add origin https://github.com/Ondiek-source/trader-ai.git
}

git push -u origin main --force

Write-Host "SUCCESS: Your GitHub repo is now lean and fast!" -ForegroundColor Green
