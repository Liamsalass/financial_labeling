@echo off

call setup_xml.bat
if %ERRORLEVEL% neq 0 (
    echo "setup_xml.bat failed"
    cd scripts
    exit /b 1
)

cd ..

python finer_preprocess.py

if %ERRORLEVEL% neq 0 (
    echo "finer_preprocess.py failed"
    cd scripts
    exit /b 1
)

python preprocess.py --text-path finer_texts.txt --tokenized-path finer_tokenized.txt --label-path finer_labels.txt --vocab-path vocab.npy --emb-path emb_init.npy --w2v-model word2vec.model --vocab-size 300 --max-len 700
if %ERRORLEVEL% neq 0 (
    echo "preprocess.py failed"
    cd scripts
    exit /b 1
)

python main.py --data-cnf configure/datasets/FINER-139.yaml --model-cnf configure/models/AttentionXML-FINER-139.yaml --mode train
if %ERRORLEVEL% neq 0 (
    echo "main.py failed"
    
    cd scripts

    exit /b 1
)

cd scripts