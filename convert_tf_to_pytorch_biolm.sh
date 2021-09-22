export BERT_BASE_DIR=/wynton/protected/project/fda_adr/msushil/bert_models

python transformers/src/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path $BERT_BASE_DIR/3/512/batch32/model.ckpt-70000 \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path $BERT_BASE_DIR/3/512/pytorch/70k/pytorch_model.bin
