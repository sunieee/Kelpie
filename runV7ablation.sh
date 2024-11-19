run() {
    model=$1
    dataset=$2
    device=$3

    output_dir="out/${model}_${dataset}"
    mkdir -p "$output_dir"

    #  1000 0100 0010 0001
    # 1110 1101 1011 0111 1000
    for ablation in 1011; do
        CUDA_VISIBLE_DEVICES=${device} python3 processV2.py --dataset "$dataset" --model "$model" --ablation "$ablation"  > "${output_dir}/process_${ablation}.log"  2>&1

        if [[ "$dataset" == "FB15k" || "$dataset" == "FB15k-237" ]]; then
            CUDA_VISIBLE_DEVICES=${device} python3 verify.py --dataset "$dataset" --model "$model" --metric eXpath --topN 4 --ablation "$ablation" > "${output_dir}/verify_eXpath(${ablation})4.log"
        else
            CUDA_VISIBLE_DEVICES=${device} python3 verify.py --dataset "$dataset" --model "$model" --metric eXpath --topN 4 --filter head --ablation "$ablation" > "${output_dir}/verify_eXpath(h${ablation})4.log"
        fi
    done
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
