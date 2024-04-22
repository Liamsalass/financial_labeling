@echo off


cd ..

python main.py --data-cnf configure/datasets/FINER-139.yaml --model-cnf configure/models/AttentionXML-FINER-139.yaml --mode train
if %ERRORLEVEL% neq 0 (
    echo "main.py failed"
    
    cd scripts

    exit /b 1
)

cd scripts