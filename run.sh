python run_classifier.py   --task_name=MRPC   --do_train=true   --do_eval=true   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=/tmp/mrpc_output2

python3  inspect_checkpoint.py  --file_name $BERT_BASE_DIR/bert_model.ckpt --all_tensors

python3  convert.py  --file_name chinese_L-12_H-768_A-12/bert_model.ckpt  --show_converted
