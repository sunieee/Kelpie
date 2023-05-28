

# dataset=MOF-3000
dataset=FB15k-237
device=1
# method=ConvE          TransE
# embedding_model=CompGCN       ""

explain() {
        embedding_model=$1
        method=$2
        run=$3  # 111
        output_folder=results/$dataset/${method}${embedding_model}
        mkdir -p $output_folder
        explain_path=$output_folder/../explain.csv
        model_path=stored_models/"${method}${embedding_model}_${dataset}.pt"

        echo $output_folder

        CUDA_VISIBLE_DEVICES=$device python explain.py --dataset $dataset --method=$method \
                --model_path $model_path --explain_path $explain_path \
                --output_folder $output_folder  --run $run --relevance_method hybrid # > $output_folder/output.log
                # --specify_relation --ignore_inverse \
                # --embedding_model "$embedding_model" --train_restrain
}

explain "" ConvE 001
# explain "" ConvE 011
# explain CompGCN ConvE 011
# explain CompGCN ConvE 011