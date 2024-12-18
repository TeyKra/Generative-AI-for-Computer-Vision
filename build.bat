@echo off
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%\venv"
set "REQUIREMENTS_FILE=%PROJECT_DIR%\requirements.txt"

IF EXIST "%VENV_DIR%" (
    echo [INFO] Erasing existing venv...
    rmdir /s /q "%VENV_DIR%"
)

echo [INFO] Creating venv...
python -m venv "%VENV_DIR%"
call "%VENV_DIR%\Scripts\activate"

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing dependencies...
python -m pip install -r "%REQUIREMENTS_FILE%"

call deactivate
echo [INFO] venv created. Press Enter to exit.
pause
