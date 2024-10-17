##############################
### END TO END EXPERIMENTS ###
##############################

python test.py --model complex --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2

# kelpie
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation /people/person/profession
python3 verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex
mv output* out/complex_FB15k-237/kelpie

# k1
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation /people/person/profession --baseline k1  
mv output* out/complex_FB15k-237/k1

# k1_abstract
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation /people/person/profession --baseline k1_abstract
mv output* out/complex_FB15k-237/k1_abstract

# k1_relation_double
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path out/complex_FB15k-237/input_facts.csv --model complex --relation /people/person/profession --baseline k1_relation_double
mv output* out/complex_FB15k-237/k1_relation_double




# Kelpie Necessary ComplEx FB15k-237
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --model complex && \
python3 verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_complex_fb15k237.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18
python3 explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --model complex && \
python3 verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --model complex && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_complex_wn18.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE WN18RR
python3 explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --model conve && \
python3 verify_explanations.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --model conve && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_conve_wn18rr.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE FB15k
python3 explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --model transe && \
python3 verify_explanations.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --model transe && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_transe_fb15k.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE YAGO3-10
python3 explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --model transe && \
python3 verify_explanations.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --model transe && \
mv output_end_to_end.csv scripts/experiments/end_to_end/kelpie_necessary_transe_yago310.csv && \
rm output_*.csv && \

##############################
### MINIMALITY EXPERIMENTS ###
##############################

# Kelpie Necessary ComplEx WN18 (Minimality)
python3 explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --model complex && \
python3 verify_explanations_skip_random.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --model complex && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_complex_wn18_sampled.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k-237 (Minimality)
python3 explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --model complex && \
python3 verify_explanations_skip_random.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --model complex && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_complex_fb15k237_sampled.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE WN18RR (Minimality)
python3 explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --model conve && \
python3 verify_explanations_skip_random.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --model conve && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_conve_wn18rr_sampled.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE FB15k (Minimality)
python3 explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --model transe && \
python3 verify_explanations_skip_random.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --model transe && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_transe_fb15k_sampled.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE YAGO3-10 (Minimality)
python3 explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --model transe && \
python3 verify_explanations_skip_random.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --model transe && \
[ ! -f output_end_to_end_skipping_random_facts.txt ] || mv output_end_to_end_skipping_random_facts.txt scripts/experiments/end_to_end/kelpie_necessary_transe_yago310_sampled.csv && \
rm output_*.csv && \
