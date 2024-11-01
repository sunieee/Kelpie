export CUDA_VISIBLE_DEVICES=0
#!/bin/bash
for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    output_dir="out/complex_${dataset}"
    mkdir -p "$output_dir"
    
    python3 explain.py --dataset "$dataset" --model complex --baseline k1_abstract --perspective double > "${output_dir}/explain.log"
    python3 process.py --dataset "$dataset" --path "$output_dir" > "${output_dir}/process.log"
    
    python3 verify.py --dataset "$dataset" --model complex --metric R --topN 4 > "${output_dir}/verify_R4.log"
    python3 verify.py --dataset "$dataset" --model complex --metric GA --topN 4 > "${output_dir}/verify_GA4.log"
    python3 verify.py --dataset "$dataset" --model complex --metric R --filter head --topN 4 > "${output_dir}/verify_R4.head.log"
done


export CUDA_VISIBLE_DEVICES=1
# ConvE
for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    output_dir="out/conve_${dataset}"
    mkdir -p "$output_dir"
    
    python3 explain.py --dataset "$dataset" --model conve --baseline k1_abstract --perspective double > "${output_dir}/explain.log"
    python3 process.py --dataset "$dataset" --path "$output_dir" > "${output_dir}/process.log"
    
    python3 verify.py --dataset "$dataset" --model conve --metric R --topN 4 > "${output_dir}/verify_R4.log"
    python3 verify.py --dataset "$dataset" --model conve --metric GA --topN 4 > "${output_dir}/verify_GA4.log"
    python3 verify.py --dataset "$dataset" --model conve --metric R --filter head --topN 4 > "${output_dir}/verify_R4.head.log"
done


# TransE
for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    output_dir="out/transe_${dataset}"
    mkdir -p "$output_dir"
    
    python3 explain.py --dataset "$dataset" --model transe --baseline k1_abstract --perspective double > "${output_dir}/explain.log"
    python3 process.py --dataset "$dataset" --path "$output_dir" > "${output_dir}/process.log"
    
    python3 verify.py --dataset "$dataset" --model transe --metric R --topN 4 > "${output_dir}/verify_R4.log"
    python3 verify.py --dataset "$dataset" --model transe --metric GA --topN 4 > "${output_dir}/verify_GA4.log"
    python3 verify.py --dataset "$dataset" --model transe --metric R --filter head --topN 4 > "${output_dir}/verify_R4.head.log"
done


# baseline 5 * 3 * 20 = 300min = 5h
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