@echo off
cd /d %~dp0
title HW2 Part I GPT Decoder FAST Training


REM Create venv only if it does not already exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting FAST training...
echo Model: d_model=64, heads=4, layers=2, block_size=64, epochs=2
echo.

python train.py --seed 42 --epochs 2 --batch_size 16 --block_size 64 --d_model 64 --n_heads 4 --n_layers 2 --d_ff 256 --lr 5e-4 --warmup_steps 20





echo.
echo Training finished. Check the outputs folder:
echo outputs\final_results.json
echo outputs\training_log.csv
echo outputs\loss_plot.png
echo outputs\best_model.pt
echo.

pause
