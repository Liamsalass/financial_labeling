@REM MobileBERT Training and Testing
call python rq3_train.py -model_name MobileBERT -output_checkpoint_path rq3_mobilebert_model -peft 1 -train_batch_size 48 -epochs 2
call python rq3_test.py -model_name MobileBERT -checkpoint_path rq3_mobilebert_model -peft 1
@REM TODO: Add plotting code here, add other 4 models
