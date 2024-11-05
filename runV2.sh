export CUDA_VISIBLE_DEVICES=0

# for model in complex conve transe; do
#     for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
#         output_dir="out/${model}_${dataset}"
#         mkdir -p "$output_dir"
#         python3 explain.py --dataset "$dataset" --model ${model} --baseline k1_abstract --perspective double  > "${output_dir}/explain.log"  2>&1
#     done
# done

# for model in complex conve transe; do
#     for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
#         output_dir="out/${model}_${dataset}"
#         mkdir -p "$output_dir"
#         python3 process.py --dataset "$dataset" --model "$model" > "${output_dir}/process.log"  2>&1
#     done
# done

for model in complex conve transe; do
    for dataset in FB15k WN18 FB15k-237 WN18RR; do
        output_dir="out/${model}_${dataset}"
        mkdir -p "$output_dir"
        python3 process.py --dataset "$dataset" --model "$model" > "${output_dir}/process.log"  2>&1
    done
done

for model in complex conve transe; do
    for dataset in FB15k WN18 FB15k-237 WN18RR; do
        output_dir="out/${model}_${dataset}"
        mkdir -p "$output_dir"

        CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset "$dataset" --model "$model" --metric score --topN 4 > "${output_dir}/verify_score4.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset "$dataset" --model "$model" --metric score --topN 4 --filter head > "${output_dir}/verify_score4.head.log" 2>&1 &
        # 等待这两个进程执行完
        wait
    done
done



python3 explain.py --dataset FB15k --model complex --baseline k1_abstract --perspective double > out/complex_FB15k/explain.log
python3 process.py --dataset FB15k --model complex > out/complex_FB15k/process.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k --model complex --metric score --topN 4 > "out/complex_FB15k/verify_score4.log"
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k --model complex --metric score --topN 4 --filter head > "out/complex_FB15k/verify_score4.head.log"