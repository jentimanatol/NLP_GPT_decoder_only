@echo off
cd /d %~dp0
title HW2 Part I GPT Decoder ULTRA FAST Training

REM ============================================================
REM ULTRA FAST VERSION
REM This runs a fixed number of training steps, not full epochs.
REM Use this on CPU when full-epoch training is too slow.
REM ============================================================

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting ULTRA FAST limited-step training...
echo.

python train_fast.py --seed 42 --max_steps 300 --eval_every 50 --eval_batches 20 --batch_size 16 --block_size 64 --d_model 64 --n_heads 4 --n_layers 2 --d_ff 256 --lr 5e-4 --warmup_steps 20

echo.
echo Finished. Check:
echo outputs\final_results.json
echo outputs\training_log.csv
echo outputs\loss_plot.png
echo outputs\best_model.pt
echo.

pause
