# Upgrade MLflow to SQLite Backend (Optional)

Write-Host "`nUpgrading MLflow to SQLite backend..." -ForegroundColor Cyan

# Stop any running MLflow servers first
Write-Host "Please stop the MLflow UI (Ctrl+C in the MLflow terminal)" -ForegroundColor Yellow
Write-Host "Press any key when ready to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Create SQLite database
Write-Host "`nCreating SQLite database..." -ForegroundColor Green
New-Item -ItemType Directory -Path "mlflow_db" -Force | Out-Null

# Start MLflow with SQLite backend
Write-Host "Starting MLflow UI with SQLite backend..." -ForegroundColor Green
Write-Host "MLflow UI will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""

mlflow ui --backend-store-uri sqlite:///mlflow_db/mlflow.db --default-artifact-root ./mlruns
