@echo off
cd /d %~dp0
title HW2 Part I GPT Decoder Training

python -m venv venv
call venv\Scripts\activate

pip install -r requirements.txt

python train.py --seed 42 --epochs 5 --batch_size 16 --block_size 128 --d_model 128 --n_heads 4 --n_layers 2 --lr 3e-4

pause
