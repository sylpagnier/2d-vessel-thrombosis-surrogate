# Rewrite git history to remove AI co-author trailers from all commit messages.
# After this completes, force-push to update GitHub:
#   git push --force-with-lease origin master
#
# WARNING: Rewrites history. Coordinate with anyone else using this repo.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:FILTER_BRANCH_SQUELCH_WARNING = "1"
git filter-branch -f --msg-filter "python scripts/git-hooks/strip_ai_coauthor_stdin.py" -- --all

Write-Host ""
Write-Host "[OK] History rewritten locally."
Write-Host "[i] Verify: git log --grep='Co-authored-by' --oneline"
Write-Host "[i] Then push: git push --force-with-lease origin master"
