@echo off

cd ..

call conda env list | findstr /C:"xml" 1>nul
if errorlevel 1 (
    echo "Creating new environment"
    call conda env create -f xml.yml
) else (
    echo "Activating existing environment"
)

call conda activate xml


echo "Preprocessing FINER train data"
::python finer_preprocess.py --output-dir "./processed" --split "train"  --vocab-size 50000 --max-len 728

echo "Preprocessing FINER test data"
::python finer_preprocess.py --output-dir "./processed" --split "test"  --vocab-size 50000 --max-len 728

echo "Converting train data to XML format"
python preprocess.py --text-path "./processed_train/texts.npy" --tokenized-path "./processed_train/tokenized.npy" --label-path "./processed_train/tags.npy" --vocab-path "./processed_train/vocab.npy" --max-len 500

echo "Converting test data to XML format"
python preprocess.py --text-path "./processed_test/texts.npy" --tokenized-path "./processed_test/tokenized.npy" --label-path "./processed_test/tags.npy" --vocab-path "./processed_test/vocab.npy"  --max-len 500

echo "All data preprocessed and ready for XML model training."

cd scripts
