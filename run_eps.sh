

device=1
# mode=necessary        sufficient
# method=ConvE          TransE
# embedding_model=CompGCN       ""

explain() {
        mode=necessary
        dataset=$1
        embedding_model=$2
        method=$3
        run=$4  # 111
        output_folder=results/$dataset/${method}${embedding_model}-${mode}
        mkdir -p $output_folder
        explain_path=$output_folder/explain.csv
        model_path=stored_models/"${method}${embedding_model}_${dataset}.pt"

        echo $output_folder

        CUDA_VISIBLE_DEVICES=$device python eps.py --dataset $dataset --method=$method \
                --model_path $model_path --explain_path $explain_path --mode $mode \
                --output_folder $output_folder  --run $run --embedding_model "$embedding_model" \
                --relation_path
                # --specify_relation --ignore_inverse --train_restrain  \
                # --prefilter_threshold 50
                # --relation_path
                # > "results/${mode}_example.log" 
}

explain FB15k-237 "" ConvE 001
# explain sufficient "" ConvE 011
# explain necessary CompGCN ConvE 011
# explain sufficient CompGCN ConvE 011