

# dataset=MOF-3000
dataset=FB15k-237
device=1
# mode=necessary        sufficient
# method=ConvE          TransE
# embedding_model=CompGCN       ""

explain() {
        mode=$1
        embedding_model=$2
        method=$3
        run=$4  # 111
        output_folder=results/$dataset/${method}${embedding_model}-${mode}-rank
        mkdir -p $output_folder
        explain_path=$output_folder/../explain.csv
        model_path=stored_models/"${method}${embedding_model}_${dataset}.pt"

        echo $output_folder

        CUDA_VISIBLE_DEVICES=$device python explain.py --dataset $dataset --method=$method \
                --model_path $model_path --explain_path $explain_path --mode $mode \
                --output_folder $output_folder  --run $run  --relevance_method rank \
                > $output_folder/$mode.log
                # --specify_relation --ignore_inverse \
                # --embedding_model "$embedding_model" --train_restrain
                # > "results/${mode}_example.log" 
}

explain necessary "" ConvE 001
# explain sufficient "" ConvE 011
# explain necessary CompGCN ConvE 011
# explain sufficient CompGCN ConvE 011