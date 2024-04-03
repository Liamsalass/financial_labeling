# Testing each model with a subset of the data, and storing results in txt files.
# Gave exec permissions with chmod +x run_rq3.sh
# 2>&1 | tee allows printing of stdout and stderr to the txt file.
subset_size=64
python3 rq3_test.py -model_name MobileBERT -subset "$subset_size" -checkpoint_path "rq3_model/checkpoint-32" 2>&1 | tee results/MobileBERT-Subset-$subset_size.txt
echo "\n\n"
python3 rq3_test.py -model_name SEC-BERT-BASE -subset "$subset_size" 2>&1 | tee results/SEC-BERT-BASE-Subset-$subset_size.txt
echo "\n\n"
python3 rq3_test.py -model_name SEC-BERT-NUM -subset "$subset_size" 2>&1 | tee results/SEC-BERT-NUM-Subset-$subset_size.txt
echo "\n\n"
python3 rq3_test.py -model_name SEC-BERT-SHAPE -subset "$subset_size" 2>&1 | tee results/SEC-BERT-SHAPE-Subset-$subset_size.txt