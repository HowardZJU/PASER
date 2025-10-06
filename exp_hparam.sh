#!/bin/bash
MAX_JOBS=16
TOTAL_GPUS=8
MAX_RETRIES=1

get_gpu_allocation() {
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=$((job_number % TOTAL_GPUS))
    echo $gpu_id+1
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0


run_with_retry() {
    local script=$1
    local gpu_allocation=$2
    local attempt=0
    while [ $attempt -le $MAX_RETRIES ]; do
        # Run the Python script
        CUDA_VISIBLE_DEVICES=$gpu_allocation python $script
        status=$?
        if [ $status -eq 0 ]; then
            echo "Script $script succeeded."
            break
        else
            echo "Script $script failed on attempt $attempt. Retrying..."
            ((attempt++))
        fi
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Script $script failed after $MAX_RETRIES attempts."
    fi
}

job_number=0


for data in Beauty Yelp; do
for full_sort in 0; do
for seed in 1 2 3 4 5; do
for lr in 0.0005 0.001 0.005 0.01 0.05 0.1; do
    ROOT=sensitivity/lr/${data}_${lr}_${seed}
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/main.py --model_name STFTW --lr $lr --data_name $data --seed $seed --output_dir $ROOT" "$gpu_allocation" &
done
done
done
done

for data in Beauty Yelp; do
for full_sort in 0; do
for seed in 1 2 3 4 5; do
for batch_size in 32 64 128 256 512 1024; do
    ROOT=sensitivity/batch_size/${data}_${batch_size}_${seed}
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/main.py --model_name STFTW --batch_size $batch_size --data_name $data --seed $seed --output_dir $ROOT" "$gpu_allocation" &
done
done
done
done

for data in Beauty Yelp; do
for full_sort in 0; do
for seed in 1 2 3 4 5; do
for max_seq_length in 20 30 40 50 60 70 80; do
    ROOT=sensitivity/max_seq_length/${data}_${max_seq_length}_${seed}
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/main.py --model_name STFTW --max_seq_length $max_seq_length --data_name $data --seed $seed --output_dir $ROOT" "$gpu_allocation" &
done
done
done
done

for data in Beauty Yelp; do
for full_sort in 0; do
for seed in 1 2 3 4 5; do
for lr in 0.0005 0.001 0.005 0.01 0.05 0.1; do
    ROOT=sensitivity_rebb_sas/lr/${data}_${lr}_${seed}
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/main.py --model_name SASRec --lr $lr --data_name $data --seed $seed --output_dir $ROOT" "$gpu_allocation" &
done
done
done
done

for data in Beauty Yelp; do
for full_sort in 0; do
for seed in 1 2 3 4 5; do
for batch_size in 32 64 128 256 512 1024; do
    ROOT=sensitivity_rebb_sas/batch_size/${data}_${batch_size}_${seed}
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/main.py --model_name SASRec --batch_size $batch_size --data_name $data --seed $seed --output_dir $ROOT" "$gpu_allocation" &
done
done
done
done

for data in Beauty Yelp; do
for full_sort in 0; do
for seed in 1 2 3 4 5; do
for max_seq_length in 20 30 40 50 60 70 80; do
    ROOT=sensitivity_rebb_sas/max_seq_length/${data}_${max_seq_length}_${seed}
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/main.py --model_name SASRec --max_seq_length $max_seq_length --data_name $data --seed $seed --output_dir $ROOT" "$gpu_allocation" &
done
done
done
done


wait
exec 6>&-
