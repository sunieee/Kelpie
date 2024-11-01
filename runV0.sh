export CUDA_VISIBLE_DEVICES=0
mkdir -p out/complex_FB15k-237
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --model complex --baseline k1_abstract --perspective double > out/complex_FB15k-237/explain.log
python3 process.py --dataset FB15k-237 --path out/complex_FB15k-237
python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --topN 5 > out/complex_FB15k-237/verify_R5.log
python3 verify.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex --metric R --filter head --topN 5 > out/complex_FB15k-237/verify_R5.head.log

mkdir -p out/complex_WN18RR
python3 explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --model complex --baseline k1_abstract --perspective double > out/complex_WN18RR/explain.log
python3 process.py --dataset WN18RR --path out/complex_WN18RR
python3 verify.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --model complex --metric R --topN 5 > out/complex_WN18RR/verify_R5.log
python3 verify.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --model complex --metric R --filter head --topN 5 > out/complex_WN18RR/verify_R5.head.log

mkdir -p out/complex_YAGO3-10
python3 explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --model complex --baseline k1_abstract --perspective double > out/complex_YAGO3-10/explain.log
python3 process.py --dataset YAGO3-10 --path out/complex_YAGO3-10
python3 verify.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --model complex --metric R --topN 5 > out/complex_YAGO3-10/verify_R5.log
python3 verify.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --model complex --metric R --filter head --topN 5 > out/complex_YAGO3-10/verify_R5.head.log


export CUDA_VISIBLE_DEVICES=1
# ConvE
mkdir -p out/conve_FB15k-237
python3 explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --model conve --baseline k1_abstract --perspective double > out/conve_FB15k-237/explain.log
python3 process.py --dataset FB15k-237 --path out/conve_FB15k-237
python3 verify.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --model conve --metric R --topN 5 > out/conve_FB15k-237/verify_R5.log
python3 verify.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --model conve --metric R --filter head --topN 5 > out/conve_FB15k-237/verify_R5.head.log

midkr -p out/conve_WN18RR
python3 explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --model conve --baseline k1_abstract --perspective double > out/conve_WN18RR/explain.log
python3 process.py --dataset WN18RR --path out/conve_WN18RR
ython3 verify.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --model conve --metric R --topN 5 > out/conve_WN18RR/verify_R5.log
python3 verify.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --model conve --metric R --filter head --topN 5 > out/conve_WN18RR/verify_R5.head.log

mkdir -p out/conve_YAGO3-10
python3 explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --model conve --baseline k1_abstract --perspective double > out/conve_YAGO3-10/explain.log
python3 process.py --dataset YAGO3-10 --path out/conve_YAGO3-10
python3 verify.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --model conve --metric R --topN 5 > out/conve_YAGO3-10/verify_R5.log
python3 verify.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --model conve --metric R --filter head --topN 5 > out/conve_YAGO3-10/verify_R5.head.log

## TransE
mkdir -p out/transe_FB15k-237
python3 explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --reg 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --model transe --baseline k1_abstract --perspective double > out/transe_FB15k-237/explain.log
python3 process.py --dataset FB15k-237 --path out/transe_FB15k-237
python3 verify.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --reg 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --model transe --metric R --topN 5 > out/transe_FB15k-237/verify_R5.log
python3 verify.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --reg 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --model transe --metric R --filter head --topN 5 > out/transe_FB15k-237/verify_R5.head.log

mkdir -p out/transe_WN18RR
python3 explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --reg 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --model transe --baseline k1_abstract --perspective double > out/transe_WN18RR/explain.log
python3 process.py --dataset WN18RR --path out/transe_WN18RR
python3 verify.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --reg 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --model transe --metric R --topN 5 > out/transe_WN18RR/verify_R5.log
python3 verify.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --reg 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --model transe --metric R --filter head --topN 5 > out/transe_WN18RR/verify_R5.head.log

mkdir -p out/transe_YAGO3-10
python3 explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --reg 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --model transe --baseline k1_abstract --perspective double > out/transe_YAGO3-10/explain.log
python3 process.py --dataset YAGO3-10 --path out/transe_YAGO3-10
python3 verify.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --reg 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --model transe --metric R --topN 5 > out/transe_WN18RR/verify_R5.log
python3 verify.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --reg 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --model transe --metric R --filter head --topN 5 > out/transe_WN18RR/verify_R5.head.log
