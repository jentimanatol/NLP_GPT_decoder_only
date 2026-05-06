@echo off
cd /d %~dp0
title GPT Decoder Quick CPU Training

REM ============================================================
REM Quick CPU training launcher for HW Part I.
REM This avoids full epochs and trains by fixed random mini-batch steps.
REM You should see heartbeat progress every 5 steps.
REM ============================================================

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist data\tinyshakespeare.txt (
    echo Dataset missing. Downloading Tiny Shakespeare...
    if not exist data mkdir data
    powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile data\tinyshakespeare.txt"
)

echo.
echo Starting quick CPU training...
echo.

python train_quick_cpu.py --steps 3200 --eval_every 20 --eval_batches 8 --batch_size 8 --block_size 48 --d_model 48 --n_heads 4 --n_layers 2 --d_ff 128 --lr 8e-4 --warmup_steps 10

if errorlevel 1 (
    echo.
    echo Training failed. Read the error above.
    pause
    exit /b 1
)

echo.
echo Training finished. Files created:
echo outputs\best_model.pt
echo outputs\config.json
echo outputs\tokenizer.json
echo outputs\training_log.csv
echo outputs\final_results.json
echo outputs\loss_plot.png
echo.

echo Testing generation now...
python generate.py --prompt "ROMEO:" --max_new_tokens 200 --method sample --temperature 0.8 --top_k 20

echo.
echo Done.
pause
