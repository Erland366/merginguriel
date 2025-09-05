EPOCHS=10
EVAL_DIR=./eval_results
MODEL_NAME=bert-base-multilingual-cased
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name ${MODEL_NAME} \
    --epochs ${EPOCHS} \
    --eval_path ${EVAL_DIR}/massive/${MODEL_NAME}_epoch${EPOCHS} \
    --push_to_hub \
    --override_results