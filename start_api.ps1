# Start API Server with authentication
$env:API_KEY = "dev-key-12345"

Write-Host "Starting Regime-Aware Prediction API Server..." -ForegroundColor Green
Write-Host "API Key: $env:API_KEY" -ForegroundColor Yellow
Write-Host "Server will start on http://127.0.0.1:8002" -ForegroundColor Cyan
Write-Host ""

uvicorn src.inference.api:app --host 127.0.0.1 --port 8002
