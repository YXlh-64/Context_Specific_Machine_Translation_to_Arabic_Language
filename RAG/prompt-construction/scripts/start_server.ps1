# Phase 3 startup script
# Run this to start the Phase 3 Prompt Construction service

Write-Host "Starting Phase 3 - Prompt Construction Service" -ForegroundColor Cyan
Write-Host "Port: 8003" -ForegroundColor Yellow

Set-Location $PSScriptRoot\..
python -m uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
