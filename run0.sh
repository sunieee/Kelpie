export CUDA_VISIBLE_DEVICES=0

for model in complex conve transe; do
    for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
        output_dir="out/${model}_${dataset}"
        mkdir -p "$output_dir"
        python3 explain.py --dataset "$dataset" --model ${model} --baseline k1_abstract --perspective double  > "${output_dir}/explain.log"  2>&1
    done
done