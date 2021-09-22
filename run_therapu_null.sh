export PATH=$PATH:~/.conda/envs/bio-lm-env-v2/bin:~/bio-lm/apex/
source /salilab/diva1/home/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate bio-lm-env-v2
export PYTHONPATH=~/bio-lm:$PYTHONPATH

TASK="theraputics"
DATADIR="data/tasks/theraputics/"
MODEL_TYPE="bert"
CHECKPOINT_DIR="/wynton/protected/project/outcome_pred/public_models/biobert-base-cased-v1.2/"
MAX_SEQ_LEN=512
run=0
#biolm-70k-512
rm /wynton/protected/project/outcome_pred/Harry_workspace/codes/bio-lm/data/tasks/theraputics/cached*
mkdir /wynton/protected/project/outcome_pred/Harry_workspace/Theraputics/bert/biobert-base-cased-$MAX_SEQ_LEN
rm -rf /wynton/protected/project/outcome_pred/Harry_workspace/Theraputics/bert/biobert-base-cased-$MAX_SEQ_LEN/*
#rm -rf /wynton/protected/project/outcome_pred/Harry_workspace/Theraputics/bert/biolm-70k-$MAX_SEQ_LEN/*
for seed in {10,20,30,40,50}; do
  (( run++ ))
  #OUTPUT_DIR="/wynton/protected/project/outcome_pred/Harry_workspace/Theraputics/bert/biolm-70k-$MAX_SEQ_LEN/run${run}/"
  OUTPUT_DIR="/wynton/protected/project/outcome_pred/Harry_workspace/Theraputics/bert/biobert-base-cased-$MAX_SEQ_LEN/run${run}/"
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
    --weight_decay 0.1\
    --do_train\
    --do_eval\
    --warmup_steps 0\
    --overwrite_output_dir \
    --overwrite_cache

#  CUDA_VISIBLE_DEVICES=$SGE_GPU python -m biolm.run_classification \
#    --task_name ${TASK}\
#    --data_dir ${DATADIR}\
#    --model_type ${MODEL_TYPE}\
#    --model_name_or_path ${CHECKPOINT_DIR}\
#    --tokenizer_name ${CHECKPOINT_DIR}\
#    --output_dir ${OUTPUT_DIR} \
#    --max_seq_length ${MAX_SEQ_LEN}\
#    --per_gpu_eval_batch_size 8\
#    --do_test\
#    --overwrite_output_dir \
#    --overwrite_cache 
done
