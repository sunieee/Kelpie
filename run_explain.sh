

# dataset=MOF-3000
device=0
# method=ConvE          TransE
# embedding_model=CompGCN       ""

explain() {
        dataset=$1
        embedding_model=""
        method=$2
        run=$3  # 111
        output_folder=stage4/$dataset/${method}${embedding_model}-framework
        
        echo $output_folder
        rm -rf $output_folder
        mkdir -p $output_folder
        
        explain_path=$output_folder/../explain.csv
        model_path=stored_models/"${method}${embedding_model}_${dataset}.pt"
        CUDA_VISIBLE_DEVICES=$device python explain.py --dataset $dataset --method=$method \
                --model_path $model_path --explain_path $explain_path \
                --output_folder $output_folder  --run $run --relevance_method score # > $output_folder/output.log
                # --specify_relation --ignore_inverse \
                # --embedding_model "$embedding_model" --train_restrain
}


explain FB15k-237 ConvE 001
# explain "" ConvE 011
# explain CompGCN ConvE 011
# explain CompGCN ConvE 011
