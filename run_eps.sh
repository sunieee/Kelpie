
device=0
# method=ConvE          TransE
# embedding_model=CompGCN       ""

explain() {
        dataset=$1
        embedding_model=$2
        method=$3
        run=$4  # 111
        output_folder=output/$dataset/${method}${embedding_model}
        mkdir -p $output_folder
        mkdir -p stored_models
        # explain_path=$output_folder/explain.csv
        explain_path=output/$dataset/explain.csv
        model_path=stored_models/"${method}${embedding_model}_${dataset}.pt"

        echo $output_folder

        CUDA_VISIBLE_DEVICES=$device python eps.py --dataset $dataset --method=$method \
                --model_path $model_path --explain_path $explain_path \
                --output_folder $output_folder  --run $run --embedding_model "$embedding_model" \
                --relation_path --prefilter_threshold 5 2>&1 > "$output_folder/out.log"
                # --specify_relation --ignore_inverse --train_restrain  \
                # --prefilter_threshold 50
                # --relation_path
}

explain FB15k-237 "" ConvE 001