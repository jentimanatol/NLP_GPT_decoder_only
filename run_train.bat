
@echo off
title GPT Decoder Training

cd /d %~dp0

echo Creating virtual environment...
python -m venv venv

call venv\Scripts\activate

echo Installing dependencies...
pip install torch

echo Running training...
python train.py

pause
