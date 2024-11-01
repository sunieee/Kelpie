export CUDA_VISIBLE_DEVICES=1

for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    for model in complex conve transe; do 
        output_dir="out/${model}_${dataset}"
        mkdir -p "$output_dir"
        for metric in kelpie criage data_poisoning k1; do
            echo "Processing ${model} ${dataset} ${metric}"
            python3 verify.py --dataset "$dataset" --model "$model" --metric "$metric" > "${output_dir}/verify_${metric}.log"
        done
    done
done

