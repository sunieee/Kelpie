##############################
### END TO END EXPERIMENTS ###
##############################

python test.py --model complex --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2

# kelpie
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation /people/person/profession
python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex
mv output* out/complex_FB15k-237/kelpie

# k1
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation /people/person/profession --baseline k1  
mv output* out/complex_FB15k-237/k1

# k1_abstract
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation /people/person/profession --baseline k1_abstract
mv output* out/complex_FB15k-237/k1_abstract

# k1_relation_double
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation /people/person/profession --baseline k1_abstract --perspective double
mv output* out/complex_FB15k-237/k1_relation_double


# k1_relation_double (all relation, choose 10)
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation all --baseline k1_abstract --perspective double
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --model complex --baseline k1_abstract --perspective double

# 测试不同策略的效果
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --topN 4 > verify_GA4.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --topN 4 > verify_R4.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric RHC --topN 4 > verify_RHC4.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric RSC --topN 4 > verify_RSC4.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric HCSC --topN 4 > verify_HCSC4.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric kelpie > verify_kelpie.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --topN 3 > verify_GA3.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --topN 5 > verify_GA5.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --topN 6 > verify_GA6.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --topN 8 > verify_GA8.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --topN 6 > verify_R6.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --topN 8 > verify_R8.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --topN 3 > verify_R3.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --topN 5 > verify_R5.log

CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R+GA --topN 2 > verify_R+GA2.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R+GA --topN 3 > verify_R+GA3.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R+SC --topN 2 > verify_R+SC2.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R+SC --topN 3 > verify_R+SC3.log

CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter head --topN 4 > verify_R4.head.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter tail --topN 4 > verify_R4.tail.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter ht --topN 4 > verify_R4.ht.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter greater --topN 4 > verify_R4.greater.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter head --topN 5 > verify_R5.head.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter tail --topN 5 > verify_R5.tail.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter ht --topN 5 > verify_R5.ht.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter greater --topN 5 > verify_R5.greater.log

CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --filter head --topN 4 > verify_GA4.head.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --filter tail --topN 4 > verify_GA4.tail.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --filter ht --topN 4 > verify_GA4.ht.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric GA --filter greater --topN 4 > verify_GA4.greater.log

CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter head --topN 6 > verify_R6.head.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter tail --topN 6 > verify_R6.tail.log
CUDA_VISIBLE_DEVICES=0 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter ht --topN 6 > verify_R6.ht.log
CUDA_VISIBLE_DEVICES=1 python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter greater --topN 6 > verify_R6.greater.log



# 不同模型 + 数据： 统一使用 R5.head 及 R5（没有限制）
## ComplEx
mkdir -p out/complex_FB15k-237
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --model complex --baseline k1_abstract --perspective double > out/complex_FB15k-237/explain.log
python3 process.py --dataset FB15k-237 --path out/complex_FB15k-237
python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --topN 5 > out/complex_FB15k-237/verify_R5.log
python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter head --topN 5 > out/complex_FB15k-237/verify_R5.head.log
mv output* out/complex_FB15k-237 && mv *.log out/complex_FB15k-237

mkdir -p out/complex_WN18RR
python3 explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --model complex --baseline k1_abstract --perspective double --output_path out/complex_WN18RR/output.json > out/complex_WN18RR/explain.log
python3 process.py --dataset WN18RR --output_path out/complex_WN18RR
python3 verify.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --model complex --metric R --topN 5 > out/complex_WN18RR/verify_R5.log
python3 verify.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --model complex --metric R --filter head --topN 5 > out/complex_WN18RR/verify_R5.head.log
mv output* out/complex_WN18RR && mv *.log out/complex_WN18RR

mkdir -p out/complex_YAGO3-10
python3 explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --model complex --baseline k1_abstract --perspective double > out/complex_YAGO3-10/explain.log
python3 process.py --dataset YAGO3-10 --path out/complex_YAGO3-10
python3 verify.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --model complex --metric R --topN 5 > out/complex_YAGO3-10/verify_R5.log
python3 verify.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --model complex --metric R --filter head --topN 5 > out/complex_YAGO3-10/verify_R5.head.log
mv output* out/complex_YAGO3-10 && mv *.log out/complex_YAGO3-10

## ConvE
mkdir -p out/conve_FB15k-237
python3 explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --model conve --baseline k1_abstract --perspective double > out/conve_FB15k-237/explain.log
python3 process.py --dataset FB15k-237 --path out/conve_FB15k-237
python3 verify.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --model conve --metric R --topN 5 > out/conve_FB15k-237/verify_R5.log
python3 verify.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --model conve --metric R --filter head --topN 5 > out/conve_FB15k-237/verify_R5.head.log
mv output* out/conve_FB15k-237 && mv *.log out/conve_FB15k-237

midkr -p out/conve_WN18RR
python3 explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --model conve --baseline k1_abstract --perspective double > out/conve_WN18RR/explain.log
python3 process.py --dataset WN18RR --path out/conve_WN18RR
ython3 verify.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --model conve --metric R --topN 5 > out/conve_WN18RR/verify_R5.log
python3 verify.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --model conve --metric R --filter head --topN 5 > out/conve_WN18RR/verify_R5.head.log
mv output* out/conve_WN18RR && mv *.log out/conve_WN18RR

mkdir -p out/conve_YAGO3-10
python3 explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --model conve --baseline k1_abstract --perspective double > out/conve_YAGO3-10/explain.log
python3 process.py --dataset YAGO3-10 --path out/conve_YAGO3-10
python3 verify.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --model conve --metric R --topN 5 > out/conve_YAGO3-10/verify_R5.log
python3 verify.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --model conve --metric R --filter head --topN 5 > out/conve_YAGO3-10/verify_R5.head.log
mv output* out/conve_YAGO3-10 && mv *.log out/conve_YAGO3-10

## TransE
mkdir -p out/transe_FB15k-237
python3 explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --reg 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --model transe --baseline k1_abstract --perspective double > out/transe_FB15k-237/explain.log
python3 process.py --dataset FB15k-237 --path out/transe_FB15k-237
python3 verify.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --reg 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --model transe --metric R --topN 5 > out/transe_FB15k-237/verify_R5.log
python3 verify.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --reg 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --model transe --metric R --filter head --topN 5 > out/transe_FB15k-237/verify_R5.head.log
mv output* out/transe_FB15k-237 && mv *.log out/transe_FB15k-237

mkdir -p out/transe_WN18RR
python3 explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --reg 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --model transe --baseline k1_abstract --perspective double > out/transe_WN18RR/explain.log
python3 process.py --dataset WN18RR --path out/transe_WN18RR
python3 verify.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --reg 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --model transe --metric R --topN 5 > out/transe_WN18RR/verify_R5.log
python3 verify.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --reg 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --model transe --metric R --filter head --topN 5 > out/transe_WN18RR/verify_R5.head.log
mv output* out/transe_WN18RR && mv *.log out/transe_WN18RR

mkdir -p out/transe_YAGO3-10
python3 explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --reg 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --model transe --baseline k1_abstract --perspective double > out/transe_YAGO3-10/explain.log
python3 process.py --dataset YAGO3-10 --path out/transe_YAGO3-10
python3 verify.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --reg 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --model transe --metric R --topN 5 > out/transe_WN18RR/verify_R5.log
python3 verify.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --reg 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --model transe --metric R --filter head --topN 5 > out/transe_WN18RR/verify_R5.head.log
mv output* out/transe_YAGO3-10 && mv *.log out/transe_YAGO3-10




# Kelpie Necessary ComplEx FB15k-237
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --model complex && \
python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_complex_fb15k237.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18
python3 explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --model complex && \
python3 verify.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --model complex && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_complex_wn18.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE WN18RR
python3 explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --model conve && \
python3 verify.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --model conve && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_conve_wn18rr.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE FB15k
python3 explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --model transe && \
python3 verify.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --model transe && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_transe_fb15k.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE YAGO3-10
python3 explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --model transe && \
python3 verify.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --model transe && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_transe_yago310.csv && \
rm output_*.csv && \

##############################
### MINIMALITY EXPERIMENTS ###
##############################

# Kelpie Necessary ComplEx WN18 (Minimality)
python3 explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --model complex && \
python3 verify_skip_random.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --model complex && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_complex_wn18_sampled.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k-237 (Minimality)
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --model complex && \
python3 verify_skip_random.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_complex_fb15k237_sampled.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE WN18RR (Minimality)
python3 explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --model conve && \
python3 verify_skip_random.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --model conve && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_conve_wn18rr_sampled.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE FB15k (Minimality)
python3 explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --model transe && \
python3 verify_skip_random.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --model transe && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_transe_fb15k_sampled.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE YAGO3-10 (Minimality)
python3 explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --model transe && \
python3 verify_skip_random.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --model transe && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_transe_yago310_sampled.csv && \
rm output_*.csv && \
