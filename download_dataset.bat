@echo off
cd /d %~dp0

if not exist data mkdir data

echo Downloading Tiny Shakespeare dataset...
powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile data\tinyshakespeare.txt"

echo Done.
echo Saved to data\tinyshakespeare.txt
pause
