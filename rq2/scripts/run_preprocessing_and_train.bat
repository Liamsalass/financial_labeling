@echo off

cd ..

python finer_preprocess.py

python preprocess.py --text-path finer_texts.txt --tokenized-path finer_tokenized.txt --label-path finer_labels.txt --vocab-path vocab.npy --emb-path emb_init.npy --w2v-model word2vec.model --vocab-size 1000 --max-len 100

python main.py --data-cnf configure/datasets/FINER-139.yaml --model-cnf configure/models/AttentionXML-FINER-139.yaml --mode train

cd scripts