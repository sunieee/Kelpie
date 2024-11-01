export CUDA_VISIBLE_DEVICES=0

for model in complex conve transe; do
    for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
        output_dir="out/complex_${dataset}"
        mkdir -p "$output_dir"
        python3 explain.py --dataset "$dataset" --model complex --baseline k1_abstract --perspective double > "${output_dir}/explain.log"
    done
done