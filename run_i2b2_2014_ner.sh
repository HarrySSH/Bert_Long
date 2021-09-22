export PATH=$PATH:~/anaconda3/bin:~/bio-lm/apex/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bio-lm-env
export PYTHONPATH=~/bio-lm:$PYTHONPATH

TASK="I2B22014"
MAX_SEQ_LEN=128
DATADIR="data/tasks/I2B22014NER.model=ucsf-bert.maxlen=${MAX_SEQ_LEN}"
MODEL_TYPE="bert"
CHECKPOINT_DIR="/wynton/protected/project/fda_adr/msushil/bert_models/3/pytorch/500k/"

run=0
for seed in {10,20,30,40,50}; do
  (( run++ ))
  OUTPUT_DIR="/wynton/protected/project/fda_adr/msushil/finetuning_models/i2b2_2014_ner/bert/biolm-500k-128/run${run}/"
  CUDA_VISIBLE_DEVICES=$SGE_GPU python -m biolm.run_sequence_labelling \
    --data_dir ${DATADIR}\
    --model_type ${MODEL_TYPE}\
    --labels ${DATADIR}/labels.txt\
    --model_name_or_path ${CHECKPOINT_DIR}\
    --tokenizer_name ${CHECKPOINT_DIR}\
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LEN}\
    --num_train_epochs 20\
    --per_gpu_train_batch_size 8\
    --per_gpu_eval_batch_size 8\
    --save_steps 500\
    --seed ${seed}\
    --gradient_accumulation_steps 4\
    --learning_rate 1e-5\
    --do_train\
    --do_eval\
    --eval_all_checkpoints\
    --overwrite_output_dir \
    --overwrite_cache

  CUDA_VISIBLE_DEVICES=$SGE_GPU python -m biolm.run_sequence_labelling \
    --data_dir ${DATADIR}\
    --model_type ${MODEL_TYPE}\
    --labels ${DATADIR}/labels.txt \
    --model_name_or_path ${CHECKPOINT_DIR}\
    --tokenizer_name ${CHECKPOINT_DIR}\
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LEN}\
    --per_gpu_eval_batch_size 8\
    --do_predict\
    --seed ${seed}
done
