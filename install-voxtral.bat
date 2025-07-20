@echo off
echo Installing Voxtral dependencies...

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install transformers from source to get latest features including Voxtral
echo Installing transformers from source...
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers.git

REM Install other required packages
echo Installing other dependencies...
pip install mistral-common[audio]
pip install librosa
pip install soundfile

echo Voxtral dependencies installed successfully!
echo You can now run start-webui.bat
pause