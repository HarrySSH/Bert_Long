export PATH=$PATH:~/anaconda3/bin:~/bio-lm/apex/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bio-lm-env
export PYTHONPATH=~/bio-lm:$PYTHONPATH

TASK="mednli"
DATADIR="data/tasks/MedNLI/"
MODEL_TYPE="bert"
CHECKPOINT_DIR="/wynton/protected/project/fda_adr/msushil/bert_models/3/pytorch/500k/"
MAX_SEQ_LEN=512
run=0
for seed in {10,20,30,40,50}; do
  (( run++ ))
  OUTPUT_DIR="/wynton/protected/project/fda_adr/msushil/finetuning_models/mednli/bert/biolm-500k-128/run${run}/"
  CUDA_VISIBLE_DEVICES=$SGE_GPU python -m biolm.run_classification \
    --task_name ${TASK}\
    --data_dir ${DATADIR}\
    --model_type ${MODEL_TYPE}\
    --model_name_or_path ${CHECKPOINT_DIR}\
    --tokenizer_name ${CHECKPOINT_DIR}\
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LEN}\
    --num_train_epochs 10\
    --per_gpu_train_batch_size 8\
    --per_gpu_eval_batch_size 8\
    --save_steps 200\
    --seed ${seed}\
    --gradient_accumulation_steps 2\
    --learning_rate 2e-5\
    --do_train\
    --do_eval\
    --warmup_steps 0\
    --overwrite_output_dir \
    --overwrite_cache

  CUDA_VISIBLE_DEVICES=$SGE_GPU python -m biolm.run_classification \
    --task_name ${TASK}\
    --data_dir ${DATADIR}\
    --model_type ${MODEL_TYPE}\
    --model_name_or_path ${CHECKPOINT_DIR}\
    --tokenizer_name ${CHECKPOINT_DIR}\
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LEN}\
    --per_gpu_eval_batch_size 8\
    --do_test\
    --overwrite_output_dir \
    --overwrite_cache 
done
