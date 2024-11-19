run() {
    model=$1
    dataset=$2
    device=$3

    output_dir="out/${model}_${dataset}"
    mkdir -p "$output_dir"

    CUDA_VISIBLE_DEVICES=${device} python3 processV2.py --dataset "$dataset" --model "$model" > "${output_dir}/process.log"  2>&1

    CUDA_VISIBLE_DEVICES=${device} python3 verify.py --dataset "$dataset" --model "$model" --metric score --topN 4 > "${output_dir}/verify_score4.log"
    CUDA_VISIBLE_DEVICES=${device} python3 verify.py --dataset "$dataset" --model "$model" --metric score --topN 4 --filter head > "${output_dir}/verify_score_h4.log"
}


max_jobs=2
current_jobs=0

runWrap() {
    model=$1
    dataset=$2
    device=$3

    echo "Running $model $dataset on device $device"
    touch out/$device
    # 随机睡眠 5-10s
    # sleep $((5 + RANDOM % 6))
    run $model $dataset $device
    echo "Finished $model $dataset on device $device"
    rm out/$device
}

get_available_device() {
    for device in {0..1}; do
        if [[ ! -f out/$device ]]; then
            touch out/$device
            echo $device
            return
        fi
    done
}

rm -f out/0
rm -f out/1

for model in complex conve transe; do
    for dataset in FB15k WN18 FB15k-237 WN18RR; do
        device=$(get_available_device)
        runWrap $model $dataset $device &
        ((current_jobs++))
        if [[ $current_jobs -ge $max_jobs ]]; then
            wait -n  # 等待任意一个进程完成
            ((current_jobs--))
        fi
    done
done

wait  # 等待所有进程完成


# python3 processV2.py --dataset FB15k --model complex > out/complex_FB15k/process.log
