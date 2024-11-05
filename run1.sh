export CUDA_VISIBLE_DEVICES=1
python3 verify.py --dataset "FB15k" --model "complex" --metric "k1" > "out/complex_FB15k/verify_k1.log"
python3 verify.py --dataset "FB15k" --model "transe" --metric "data_poisoning" > "out/complex_FB15k/verify_data_poisoning.log"


# for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
#     for model in complex conve transe; do 
#         output_dir="out/${model}_${dataset}"
#         mkdir -p "$output_dir"
#         for metric in kelpie criage data_poisoning k1; do
#             echo "Processing ${model} ${dataset} ${metric}"
#             python3 verify.py --dataset "$dataset" --model "$model" --metric "$metric" > "${output_dir}/verify_${metric}.log"
#         done
#     done
# done



for dataset in YAGO3-10; do
    for model in transe; do 
        output_dir="out/${model}_${dataset}"
        mkdir -p "$output_dir"
        for metric in kelpie criage data_poisoning k1; do
            echo "Processing ${model} ${dataset} ${metric}"
            python3 verify.py --dataset "$dataset" --model "$model" --metric "$metric" > "${output_dir}/verify_${metric}.log"
        done
    done
done

