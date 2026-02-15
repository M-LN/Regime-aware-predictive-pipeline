# Quick Data Location Viewer - Simple Version

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  REGIME AI - DATA AND PREDICTION LOCATIONS" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Raw Data
Write-Host "[1] RAW DATA" -ForegroundColor Yellow
Write-Host "    Location: data\raw\" -ForegroundColor White
if (Test-Path "data\raw") {
    $rawFiles = Get-ChildItem data\raw -Recurse -File | Select-Object -First 5
    if ($rawFiles.Count -gt 0) {
        foreach ($file in $rawFiles) {
            Write-Host "    - $($file.FullName)" -ForegroundColor Gray
        }
    } else {
        Write-Host "    (No files yet - data will appear here after ingestion)" -ForegroundColor Gray
    }
}
Write-Host ""

# 2. Trained Models
Write-Host "[2] TRAINED MODELS" -ForegroundColor Yellow
Write-Host "    Location: data\models\" -ForegroundColor White
if (Test-Path "data\models") {
    $models = Get-ChildItem data\models -File
    if ($models.Count -gt 0) {
        foreach ($model in $models) {
            $sizeMB = [math]::Round($model.Length / 1MB, 2)
            Write-Host "    [OK] $($model.Name) - Size: $sizeMB MB" -ForegroundColor Green
        }
    } else {
        Write-Host "    (No models trained yet)" -ForegroundColor Gray
    }
}
Write-Host ""

# 3. MLflow Experiments
Write-Host "[3] MLFLOW EXPERIMENT TRACKING" -ForegroundColor Yellow
Write-Host "    Location: mlruns\" -ForegroundColor White
if (Test-Path "mlruns") {
    $experiments = Get-ChildItem mlruns -Directory | Where-Object { $_.Name -match '^\d+$' -and $_.Name -ne "0" }
    if ($experiments.Count -gt 0) {
        Write-Host "    Experiments tracked: $($experiments.Count)" -ForegroundColor Green
        Write-Host "    To view: Run 'mlflow ui' then open http://localhost:5000" -ForegroundColor Cyan
    } else {
        Write-Host "    (No experiments yet)" -ForegroundColor Gray
    }
}
Write-Host ""

# 4. How to View Predictions
Write-Host "[4] HOW TO VIEW PREDICTIONS" -ForegroundColor Yellow
Write-Host ""
Write-Host "    OPTION 1: Interactive API Documentation (RECOMMENDED)" -ForegroundColor White
Write-Host "    -> http://127.0.0.1:8002/docs" -ForegroundColor Cyan
Write-Host "       Click on /predict, then 'Try it out'" -ForegroundColor Gray
Write-Host ""
Write-Host "    OPTION 2: Run Test Script" -ForegroundColor White
Write-Host "    -> python test_prediction.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "    OPTION 3: Quick Dashboard" -ForegroundColor White
Write-Host "    -> Open api_dashboard.html in your browser" -ForegroundColor Cyan
Write-Host ""

# 5. System Status
Write-Host "[5] SYSTEM STATUS" -ForegroundColor Yellow
Write-Host "    API Server:   http://127.0.0.1:8002" -ForegroundColor White
Write-Host "    Health Check: http://127.0.0.1:8002/health" -ForegroundColor Cyan
Write-Host "    Status:       http://127.0.0.1:8002/status" -ForegroundColor Cyan
Write-Host "    Metrics:      http://127.0.0.1:8002/metrics" -ForegroundColor Cyan
Write-Host ""

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  For complete documentation, open: USER_GUIDE.md" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
