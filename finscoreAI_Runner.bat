@echo off
setlocal EnableExtensions EnableDelayedExpansion
title FinScore AI Launcher

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
set "VENV_PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "CHECK_ONLY=0"
set "FORCE_SETUP=0"
set "MLFLOW_PORT=4022"
set "API_PORT=3022"
set "FRONTEND_PORT=2022"

if /I "%~1"=="--check" set "CHECK_ONLY=1"
if /I "%~1"=="--setup" set "FORCE_SETUP=1"

echo.
echo ============================================================
echo   FinScore AI - full project launcher
echo ============================================================
echo   Project: %PROJECT_ROOT%
echo.

if not exist "%VENV_PY%" (
    echo [1/6] Creating Python virtual environment...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv "%PROJECT_ROOT%\.venv"
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Python 3.11 or newer was not found in PATH.
            echo         Install Python from https://www.python.org/downloads/
            pause
            exit /b 1
        )
        python -m venv "%PROJECT_ROOT%\.venv"
    )
    if errorlevel 1 goto :failed
) else (
    echo [1/6] Virtual environment found.
)

"%VENV_PY%" -c "import sys; raise SystemExit(sys.version_info[:2] not in [(3, n) for n in range(11, 20)])" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] FinScore AI requires Python 3.11 or newer.
    pause
    exit /b 1
)

echo [2/6] Checking Python dependencies...
if "%FORCE_SETUP%"=="1" goto :install_dependencies
"%VENV_PY%" -c "import fastapi, uvicorn, pydantic, sqlalchemy, streamlit, pandas, numpy, sklearn, xgboost, lightgbm, shap, mlflow, joblib, plotly, requests" >nul 2>&1
if not errorlevel 1 goto :dependencies_ready

:install_dependencies
echo       Installing requirements. This can take several minutes on first run...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

:dependencies_ready
echo       Dependencies ready.

echo [3/6] Checking training dataset...
if not exist "data\german_credit.csv" (
    echo       Dataset missing; downloading OpenML credit-g v1...
    "%VENV_PY%" "data\fetch_dataset.py"
    if errorlevel 1 goto :failed
) else (
    echo       Dataset ready.
)

echo [4/6] Checking model artefacts...
"%VENV_PY%" -c "import joblib; m=joblib.load(r'models\best_model.joblib'); c=joblib.load(r'models\feature_cols.joblib'); s=joblib.load(r'models\scaler.joblib'); assert c and s.n_features_in_ == len(c) and hasattr(m, 'predict_proba')" >nul 2>&1
if errorlevel 1 (
    echo       Artefacts missing or incompatible; training models now...
    pushd "ml_pipeline"
    "%VENV_PY%" "train.py"
    if errorlevel 1 (
        popd
        goto :failed
    )
    popd
) else (
    echo       Model, feature columns, and scaler ready.
)

if "%CHECK_ONLY%"=="1" (
    echo [5/6] Running automated checks...
    "%VENV_PY%" -m pytest tests\test_preprocessing.py tests\test_risk_engine.py tests\test_api_unit.py -q
    if errorlevel 1 goto :failed
    echo [6/6] Check completed successfully. Services were not started.
    exit /b 0
)

echo [5/6] Checking ports and starting services...
set "START_MLFLOW=0"
set "START_API=0"
set "START_FRONTEND=0"

powershell -NoProfile -Command "$l=Get-NetTCPConnection -LocalPort 4022 -State Listen -ErrorAction SilentlyContinue; if (-not $l) { exit 1 }; try { $null=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:4022/health' -TimeoutSec 2; exit 0 } catch { exit 2 }" >nul 2>&1
set "MLFLOW_STATE=!ERRORLEVEL!"
if "!MLFLOW_STATE!"=="2" (
    for /f %%P in ('powershell -NoProfile -Command "foreach ($p in 4023..4099) { if (-not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) { Write-Output $p; break } }"') do set "MLFLOW_PORT=%%P"
    if "!MLFLOW_PORT!"=="4022" (
        echo [ERROR] No free MLflow port was found from 4023 through 4099.
        goto :failed
    )
    echo       Port 4022 belongs to another service; using !MLFLOW_PORT! for MLflow.
    set "START_MLFLOW=1"
)
if "!MLFLOW_STATE!"=="1" set "START_MLFLOW=1"
if "!MLFLOW_STATE!"=="0" (
    echo       Port 4022 is already in use; reusing the existing service.
)

powershell -NoProfile -Command "$l=Get-NetTCPConnection -LocalPort 3022 -State Listen -ErrorAction SilentlyContinue; if (-not $l) { exit 1 }; try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:3022/health' -TimeoutSec 2; if ($null -ne $r.ready) { exit 0 } else { exit 2 } } catch { exit 2 }" >nul 2>&1
set "API_STATE=!ERRORLEVEL!"
if "!API_STATE!"=="2" (
    for /f %%P in ('powershell -NoProfile -Command "foreach ($p in 3023..3099) { if (-not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) { Write-Output $p; break } }"') do set "API_PORT=%%P"
    if "!API_PORT!"=="3022" (
        echo [ERROR] No free API port was found from 3023 through 3099.
        goto :failed
    )
    echo       Port 3022 belongs to another service; using !API_PORT! for the API.
    set "START_API=1"
)
if "!API_STATE!"=="1" set "START_API=1"
if "!API_STATE!"=="0" (
    echo       Port 3022 is already in use; reusing the existing service.
)

powershell -NoProfile -Command "$l=Get-NetTCPConnection -LocalPort 2022 -State Listen -ErrorAction SilentlyContinue; if (-not $l) { exit 1 }; try { $null=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:2022/_stcore/health' -TimeoutSec 2; exit 0 } catch { exit 2 }" >nul 2>&1
set "FRONTEND_STATE=!ERRORLEVEL!"
if "!FRONTEND_STATE!"=="2" (
    for /f %%P in ('powershell -NoProfile -Command "foreach ($p in 2023..2099) { if (-not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) { Write-Output $p; break } }"') do set "FRONTEND_PORT=%%P"
    if "!FRONTEND_PORT!"=="2022" (
        echo [ERROR] No free dashboard port was found from 2023 through 2099.
        goto :failed
    )
    echo       Port 2022 belongs to another service; using !FRONTEND_PORT! for Streamlit.
    set "START_FRONTEND=1"
)
if "!FRONTEND_STATE!"=="1" set "START_FRONTEND=1"
if "!FRONTEND_STATE!"=="0" (
    echo       Port 2022 is already in use; reusing the existing service.
)

set "DATABASE_URL=sqlite:///./finscore_local.db"
set "API_URL=http://127.0.0.1:%API_PORT%"
set "MLFLOW_TRACKING_URI=http://127.0.0.1:%MLFLOW_PORT%"

if "%START_MLFLOW%"=="1" start "FinScore AI - MLflow" cmd /k ""%VENV_PY%" -m mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 127.0.0.1 --port %MLFLOW_PORT%"
if "%START_API%"=="1" start "FinScore AI - API" cmd /k ""%VENV_PY%" -m uvicorn backend.main:app --host 127.0.0.1 --port %API_PORT%"
if "%START_FRONTEND%"=="1" start "FinScore AI - Dashboard" cmd /k ""%VENV_PY%" -m streamlit run frontend\app.py --server.headless true --browser.gatherUsageStats false --server.address 127.0.0.1 --server.port %FRONTEND_PORT%"

echo [6/6] Waiting for API, MLflow, and dashboard readiness...
powershell -NoProfile -Command "$end=(Get-Date).AddSeconds(90); do { $api=$false; $mlflow=$false; $ui=$false; try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:%API_PORT%/ready' -TimeoutSec 2; $api=[bool]$r.ready } catch {}; try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%MLFLOW_PORT%/health' -TimeoutSec 2; $mlflow=$r.StatusCode -eq 200 } catch {}; try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%FRONTEND_PORT%/_stcore/health' -TimeoutSec 2; $ui=$r.StatusCode -eq 200 } catch {}; if ($api -and $mlflow -and $ui) { exit 0 }; Start-Sleep -Seconds 1 } while ((Get-Date) -lt $end); exit 1"
if errorlevel 1 (
    echo [WARNING] Not all FinScore services became ready within 90 seconds.
    echo           Review the three FinScore AI service windows for details.
    pause
    exit /b 1
)

echo.
echo FinScore AI is ready:
echo   Dashboard : http://127.0.0.1:%FRONTEND_PORT%
echo   API docs  : http://127.0.0.1:%API_PORT%/docs
echo   MLflow    : http://127.0.0.1:%MLFLOW_PORT%
echo.
echo Close the three service windows to stop the project.
if not "%FINSCORE_NO_BROWSER%"=="1" start "" "http://127.0.0.1:%FRONTEND_PORT%"
exit /b 0

:failed
echo.
echo [ERROR] FinScore AI setup or startup failed.
echo         Review the message above, then run this launcher again.
pause
exit /b 1
