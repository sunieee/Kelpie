

device=1
# method=ConvE          TransE
# embedding_model=CompGCN       ""

explain() {
        dataset=$1
        embedding_model=$2
        method=$3
        run=$4  # 111
        output_folder=results/$dataset/${method}${embedding_model}-necessary
        mkdir -p $output_folder
        explain_path=$output_folder/explain.csv
        model_path=stored_models/"${method}${embedding_model}_${dataset}.pt"

        echo $output_folder

        CUDA_VISIBLE_DEVICES=$device python eps.py --dataset $dataset --method=$method \
                --model_path $model_path --explain_path $explain_path \
                --output_folder $output_folder  --run $run --embedding_model "$embedding_model" \
                --relation_path --prefilter_threshold 5
                # --specify_relation --ignore_inverse --train_restrain  \
                # --prefilter_threshold 50
                # --relation_path
                # > "results/example.log" 
}

explain FB15k-237 "" ConvE 101