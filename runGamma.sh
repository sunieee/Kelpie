run() {
    model=$1
    dataset=$2
    device=$3
    gamma=$4

    output_dir="out(gamma=${gamma})/${model}_${dataset}"
    mkdir -p "$output_dir"

    CUDA_VISIBLE_DEVICES=${device} python3 process.py --dataset "$dataset" --model "$model" > "${output_dir}/process.log"  2>&1

    rm -f ${output_dir}/verify_score*4.log
    rm -f ${output_dir}/output_end_to_end_score*4.json
    # dataset = FB15k, FB15k-237
    # if [[ "$dataset" == "FB15k" ]] || [[ "$dataset" == "FB15k-237" ]]; then
    CUDA_VISIBLE_DEVICES=${device} python3 verifyGamma.py --dataset "$dataset" --model "$model" --metric score --topN 4 --gamma "$gamma" > "${output_dir}/verify_score4.log"
    # else
    CUDA_VISIBLE_DEVICES=${device} python3 verifyGamma.py --dataset "$dataset" --model "$model" --metric score --topN 4 --filter head --gamma "$gamma" > "${output_dir}/verify_score_h4.log"
    # fi
    
}


max_jobs=2
current_jobs=0

runWrap() {
    model=$1
    dataset=$2
    device=$3
    gamma=$4

    echo "Running $model $dataset on device $device (gamma=$gamma)"
    touch "out(gamma=${gamma})/$device"
    # 随机睡眠 5-10s
    # sleep $((5 + RANDOM % 6))
    run $model $dataset $device $gamma
    echo "Finished $model $dataset on device $device (gamma=$gamma)"
    rm "out(gamma=${gamma})/$device"
}

get_available_device() {
    gamma=$1
    for device in {0..1}; do
        if [[ ! -f "out(gamma=${gamma})/$device" ]]; then
            touch "out(gamma=${gamma})/$device"
            echo $device
            return
        fi
    done
}



for gamma in 0.1 0.5 1.0; do
    echo "verify with gamma=${gamma}"
    rm -f "out(gamma=${gamma})/0"
    rm -f "out(gamma=${gamma})/1"
    for model in complex conve transe; do
        for dataset in FB15k WN18 FB15k-237 WN18RR; do
            device=$(get_available_device $gamma)
            runWrap $model $dataset $device $gamma &
            ((current_jobs++))
            if [[ $current_jobs -ge $max_jobs ]]; then
                wait -n  # 等待任意一个进程完成
                ((current_jobs--))
            fi
        done
    done

    wait  # 等待所有进程完成
done


# runWrap transe FB15k-237 0 0.1
