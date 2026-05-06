@echo off
cd /d %~dp0
title HW2 Part I GPT Decoder - Generate Samples Only

REM ============================================================
REM run_generation.bat
REM Use this AFTER training is finished.
REM It tests generation only and saves the outputs.
REM ============================================================

echo.
echo ============================================================
echo GPT Decoder Generation Test
echo ============================================================
echo.

if not exist venv (
    echo ERROR: venv folder not found.
    echo Run training first or create the virtual environment.
    pause
    exit /b
)

call venv\Scripts\activate

if not exist outputs\best_model.pt (
    echo ERROR: outputs\best_model.pt not found.
    echo You must train the model first.
    pause
    exit /b
)

if not exist outputs\config.json (
    echo ERROR: outputs\config.json not found.
    echo You must train the model first.
    pause
    exit /b
)

if not exist outputs\tokenizer.json (
    echo ERROR: outputs\tokenizer.json not found.
    echo You must train the model first.
    pause
    exit /b
)

echo Running generation examples...
echo.

echo HW2 Part I GPT Decoder Generation Samples > outputs\generation_samples.txt
echo ============================================================ >> outputs\generation_samples.txt
echo. >> outputs\generation_samples.txt

echo Example 1: Greedy decoding, prompt ROMEO:
echo [Example 1: Greedy decoding, prompt ROMEO:] >> outputs\generation_samples.txt
python generate.py --prompt "ROMEO:" --method greedy --max_new_tokens 300
python generate.py --prompt "ROMEO:" --method greedy --max_new_tokens 300 >> outputs\generation_samples.txt
echo. >> outputs\generation_samples.txt
echo ------------------------------------------------------------ >> outputs\generation_samples.txt
echo. >> outputs\generation_samples.txt

echo.
echo Example 2: Sampling, prompt ROMEO:
echo [Example 2: Sampling, prompt ROMEO:, temperature 0.8, top_k 20] >> outputs\generation_samples.txt
python generate.py --prompt "ROMEO:" --method sample --temperature 0.8 --top_k 20 --max_new_tokens 300
python generate.py --prompt "ROMEO:" --method sample --temperature 0.8 --top_k 20 --max_new_tokens 300 >> outputs\generation_samples.txt
echo. >> outputs\generation_samples.txt
echo ------------------------------------------------------------ >> outputs\generation_samples.txt
echo. >> outputs\generation_samples.txt

echo.
echo Example 3: Sampling, prompt To be
echo [Example 3: Sampling, prompt To be, temperature 0.9, top_k 40] >> outputs\generation_samples.txt
python generate.py --prompt "To be" --method sample --temperature 0.9 --top_k 40 --max_new_tokens 300
python generate.py --prompt "To be" --method sample --temperature 0.9 --top_k 40 --max_new_tokens 300 >> outputs\generation_samples.txt
echo. >> outputs\generation_samples.txt
echo ------------------------------------------------------------ >> outputs\generation_samples.txt
echo. >> outputs\generation_samples.txt

echo.
echo Done.
echo Saved generation samples to:
echo outputs\generation_samples.txt
echo.
echo You can copy these samples into the report.
echo.

pause
