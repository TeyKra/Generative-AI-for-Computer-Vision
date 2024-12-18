@echo off
echo [INFO] Setting up the virtual environment...
python -m venv venv

echo [INFO] Activating the virtual environment...
call venv\Scripts\activate

echo [INFO] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo [INFO] Virtual environment setup complete!
pause
