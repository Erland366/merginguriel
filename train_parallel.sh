#!/bin/bash

EPOCHS=10
EVAL_DIR=./eval_results
MODEL_NAME=bert-base-multilingual-cased

# List of languages to train
LANGUAGES=(
    "ar-SA" "hy-AM" "bn-BD" "my-MM" "zh-CN" "zh-TW" "en-US" "fi-FI" "fr-FR"
    "ka-GE" "de-DE" "el-GR" "hi-IN" "hu-HU" "is-IS" "id-ID" "ja-JP" "jv-ID"
    "ko-KR" "lv-LV" "pt-PT" "ru-RU" "es-ES" "vi-VN" "tr-TR"
)

# Create a directory for logs
mkdir -p parallel_logs

# Function to train a single language
train_language() {
    local lang=$1
    local gpu_id=$2
    local log_file="parallel_logs/train_${lang}_gpu${gpu_id}.log"
    
    echo "Starting training for ${lang} on GPU ${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
        --model_name ${MODEL_NAME} \
        --epochs ${EPOCHS} \
        --eval_path ${EVAL_DIR}/massive/${MODEL_NAME}_${lang}_epoch${EPOCHS} \
        --push_to_hub \
        --single_lang ${lang} \
        --out_path ./models/${MODEL_NAME}_${lang} \
        > ${log_file} 2>&1 &
    
    local pid=$!
    echo "Training ${lang} started (PID: ${pid}), log: ${log_file}"
    return $pid
}

# Track running jobs
declare -a gpu0_jobs
declare -a gpu1_jobs

# Function to wait for an available GPU slot
wait_for_gpu() {
    local gpu_id=$1
    
    if [ $gpu_id -eq 0 ]; then
        # Wait for GPU 0 to have less than 1 job running
        while [ ${#gpu0_jobs[@]} -gt 0 ]; do
            # Check if any GPU 0 jobs are still running
            for i in "${!gpu0_jobs[@]}"; do
                if ! kill -0 ${gpu0_jobs[i]} 2>/dev/null; then
                    # Job finished, remove it
                    unset gpu0_jobs[i]
                    echo "GPU 0 job finished, slot available"
                fi
            done
            # Clean up array
            gpu0_jobs=("${gpu0_jobs[@]}")
            sleep 5
        done
    else
        # Wait for GPU 1 to have less than 1 job running
        while [ ${#gpu1_jobs[@]} -gt 0 ]; do
            # Check if any GPU 1 jobs are still running
            for i in "${!gpu1_jobs[@]}"; do
                if ! kill -0 ${gpu1_jobs[i]} 2>/dev/null; then
                    # Job finished, remove it
                    unset gpu1_jobs[i]
                    echo "GPU 1 job finished, slot available"
                fi
            done
            # Clean up array
            gpu1_jobs=("${gpu1_jobs[@]}")
            sleep 5
        done
    fi
}

# Counter for GPU assignment
gpu_counter=0

# Loop through languages and start training jobs
for lang in "${LANGUAGES[@]}"; do
    # Assign GPU (0 or 1)
    gpu_id=$((gpu_counter % 2))
    
    # Wait for an available slot on the assigned GPU
    wait_for_gpu $gpu_id
    
    # Train the language
    pid=$(train_language ${lang} ${gpu_id})
    
    # Add to the appropriate GPU job list
    if [ $gpu_id -eq 0 ]; then
        gpu0_jobs+=($pid)
    else
        gpu1_jobs+=($pid)
    fi
    
    # Increment counter
    ((gpu_counter++))
    
    echo "Current jobs - GPU 0: ${#gpu0_jobs[@]}, GPU 1: ${#gpu1_jobs[@]}"
done

echo "All training jobs started. Waiting for completion..."

# Wait for all remaining jobs to finish
for pid in "${gpu0_jobs[@]}" "${gpu1_jobs[@]}"; do
    if [ -n "$pid" ]; then
        wait $pid
        echo "Job $pid completed"
    fi
done

echo "All training jobs completed!"