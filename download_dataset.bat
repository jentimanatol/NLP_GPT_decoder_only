
@echo off

cd /d %~dp0

if not exist data mkdir data

powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile data\tinyshakespeare.txt"

echo Dataset downloaded.
pause
