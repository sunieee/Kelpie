<meta name="robots" content="noindex">

# Kelpie
Model-agnostic framework for explaining Link Predictions on Knowledge Graphs

<p align="center">
<img width="50%" alt="kelpie_logo" src="https://user-images.githubusercontent.com/6909990/124291133-87fa5d80-db54-11eb-9db9-62ca9bd3fe6f.png">
</p>

Kelpie is a post-hoc local explainability tool specifically tailored for embedding-based Link Prediction (LP) models on Knowledge Graphs (KGs).
It provides a simple and effective interface to identify the most relevant facts to any prediction.

## Environment and Prerequisites
We successfully run Kelpie on an Ubuntu 18.04.5 environment using used Python 3.7.7, CUDA Version: 11.2 and Driver Version: 460.73.01.
Kelpie requires the following libraries: 
- PyTorch (we used version 1.7.1);
- numpy;
- tqdm;

## Models and Datasets
Kelpie supports all LP models based on embeddings.
We run experiments on `ComplEx`, `ConvE` and `TransE` models following the implementation in this repository, explaining their predictions on datasets `FB15k`, `WN18`, `FB15k-237`, `WN18RR` and `YAGO3-10`.
The training, validation and test sets of such datasets are distributed in this repository in the `data` folder.

## Training and testing models
For the sake of reproducibility, we make available through FigShare [the `.pt` model files](https://figshare.com/s/ede27f3440fe742de60b) resulting from training each system on each dataset.
We use the following hyperparameters, which we have found to achieve the best performances.

<p align="center">
<img width="100%" alt="hyperparams" src="https://user-images.githubusercontent.com/6909990/124291956-66e63c80-db55-11eb-9aa6-9892ee24afc2.png">
</p>

Where 
* *D* is the embedding dimension (in the models we use, entity and relation embeddings always have same dimension);
* *LR* is the learning rate;
* *B* is the batch size;
* *Ep* is the number of epochs;
* *γ* is the margin in Pairwise Ranking Loss;
* *N* is the number of negative samples generated for each positive training sample;
* *Opt* is the used Optimizer (either `SGD`, or `Adagrad`, or `Adam`);
* *Reg* is the regularizer weight;
* *Decay* is the applied learning rate Decay;
* *ω* is the size of the convolutional kernels;
* *Drop* is the training dropout rate:
    * *in* is the input dropout;
    * *h* is the dropout applied after a hidden layer;
    * *feat* is the feature dropout;

After the models have been trained, evaluating them yields the following metrics:

<p align="center">
<img width="60%" alt="model_results" src="https://user-images.githubusercontent.com/6909990/135614004-db1cff3a-68db-447d-bb9c-3c7f05426957.png">
</p>

The training and evaluation processes can be launched with the commands reported in [the training and testing section](#training-and-testing-models-1)



## Running Kelpie

We showcase the effectiveness of Kelpie explaining the tail predictions of a set of 100 correctly predicted test facts for each model and dataset, both in *necessary* and in *sufficient* scenario.
The `.csv` files containing the facts we explain for each model and dataet can be found in the `input_facts` folder.
We report result files for each model and dataset, both extracted by Kelpie and by the baseline approaches, both in the necessary and for the sufficient scenario, in the `results.zip` archive in our repository.

Across the necessary and sufficient scenarios for the same model and dataset we usually employ the same set of 100 correctly predicted test facts. 
The only exception at this regard is in the `ConvE` explanations for `FB15k` and `FB15k-237` predictions: in these cases, a few <h, r, t> predictions used in necessary explanations could not be explained sufficiently because, due to strong dataset biases, *all entities in the dataset would be predicted correctly if used instead of h*.
This made it impossible to extract a set of `c` entities to convert, because any entity would appear already converted without applying any sufficient explanation.
We thus replaced these predictions for the sufficient scenario, obtaining created two different version `nec` and `_suff` for the input facts file.

Kelpie relies on a *Post-Training technique* (PT) to generate mimics and compute the relevance of potential explanations.
Across all models and datasets, we always use for the Post-Training the same combination of hyperparameters used in the original training.
The only exception is `TransE`, where the batch size *B* is particularly large (2048, which usually exceeds by far the number of training facts featuring an entity).
This affects the Post-Training process because in any Post-Training epoch the entity would only benefit from one optimization step in each epoch.
We easily balance this by increasing the learning rate *LR* to 0.01 in TransE Post-Trainings.
In order to replicate our experiments, the `.pt` model files should be located in a `stored_models` folder inside Kelpie.

Kelpie experiments are based on the execution of two separate scripts:
* an `explain.py` script that: 
   * extracts explanations and saves the top 10 relevant ones for each input prediction in an `output.txt` file;
   * logs the result each tested explanation into separate files `output_details_1.csv`;
* a `verify_explanations.py` that 
   * retrains the model from scratch after applying those explanations (i.e., removing their facts from the training set in the necessary scenario, or injecting them to the entities to convert in the sufficient scenario);
   * saves the outcomes of the corresponding predictions in an `output_end_to_end.csv` file;

The `explain.py` also accepts an optional `--baseline` parameter with allowed values `data_poisoning` or `criage`; using this parameter allows to estract results for our baselines instead of Kelpie.

Our end-to-end results for necessary explanations are the following. We measure the decrease in H@1 and MRR that we experience when removing the explanation facts and retraining the models, compared to the original metrics; the greater the decrease, the better the explanation (i.e., the "more necessary" the explanation facts actually were).

<p align="center">
<img width="60%" alt="kelpie_logo" src="https://user-images.githubusercontent.com/6909990/135614246-b68adc39-c771-404d-bc74-4ca532f8258e.png">
</p>

Our end-to-end results for sufficient explanations are the following. We add the explanation facts to 10 random entities and verify if they now display the same prediction as the one to explain, i.e., if they have been converted. In practice, we measure the explanation effectiveness by checking the increase in H@1 and MRR of the "predictions to convert", (i.e., the facts that we hope the system now predicts) after adding the explanation facts and retraining, compared to their metrics in the original model; the greater the increase, the better the explanation (i.e., the "more sufficient" the explanation facts actually were to convert those entities).

<p align="center">
<img width="60%" alt="kelpie_logo" src="https://user-images.githubusercontent.com/6909990/135614254-172bc8a1-8f58-4c6f-a84d-8c4f4e50bbde.png">
</p>

Our experiments on each model and dataset can be replicated with the commands reported in [our section on extracting and verifying explanations](#training-and-testing-models-1)
## Additional Experiments

### Prefiltering: _k_ value
The Kelpie pre-filtering module is used at the beginning of the explanation extraction to identify the most promising facts with respect to the prediction to explain. Its purpose is to narrow down the research space to the top _k_ most promising facts, thus making the research more feasible.
In all the end-to-end experiments we use _k_ = 20; we show here the effect of varying the value of _k_ on the explanations for the ComplEx model predictions:

<p align="center">
<img width="60%" alt="kelpie_logo" src="https://user-images.githubusercontent.com/6909990/135614960-e146b76a-fd99-44fa-a4a2-7a0f2f2efe62.png">
</p>

Across all datasets, varying _k_ does not cause huge variations. This suggests that the topology-based policy used by the pre-filter does indeed identify the most promising facts: in other words, if the pre-filter is good at fiding and placing the most promising facts at the top of its ranking, it does not matter if we analyze the top 10, 20 or 30 facts in the ranking: the facts that we need will still make the cut.

In WN18, WN18RR and YAGO3-10 the effect of varying _k_ seems rather negligible (as a matter of fact, they are so small they may be more tied to the inherent randomness of the embedding training process than to the variation of _k_). This can be explained by considering that in these datasets, entities are usually featured in just a feew training facts: in average, 6.9 in WN18, 4.3 IN WN18RR and 17.5 in YAGO3-10.
In FB15k and FB15k-237, on the contrary, entities tend to have far more mentions in training (in average, 64.5 in FB15k and 37.4 in FB15k-237): this makes the choice of the value of _k_ more relevant, so and varying it leads to slightly more visible consequences. In this cases, as a general rule, greater values of _k_ lead to better or equal explanations, at the cost of extending the research space. 

We do not witness any cases in which increasing _k_ to more than 20 leads to astounding improvements in the explanation relevance: this confirms that 20 is indeed a fine trade-off for the pre-filter value. 



## Launching Kelpie

### Training and testing models

The training and evaluation processes can be launched with the following commands:

* **ComplEx**

  * **FB15k**

    ```python
    python3 scripts/complex/train.py --dataset FB15k --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3
    ```
    ```python
    python3 scripts/complex/test.py --dataset FB15k --dimension 2000 --learning_rate 0.01 --model_path stored_models/ComplEx_FB15k.pt
    ```

  * **WN18**
  
    ```python
    python3 scripts/complex/train.py --dataset WN18 --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2
    ```
    ```python
    python3 scripts/complex/test.py --dataset WN18 --dimension 500 --learning_rate 0.1 --model_path stored_models/ComplEx_WN18.pt
    ```

  * **FB15k-237**
  
    ```python
    python3 scripts/complex/train.py --dataset FB15k-237 --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2
    ```
    ```python
    python3 scripts/complex/test.py --dataset FB15k-237 --dimension 1000 --learning_rate 0.1 --model_path stored_models/ComplEx_FB15k-237.pt
    ```

  * **WN18RR**
  
    ```python
    python3 scripts/complex/train.py --dataset WN18RR --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --valid 20
    ```
    ```python
    python3 scripts/complex/test.py --dataset WN18RR --dimension 500 --learning_rate 0.1 --model_path stored_models/ComplEx_WN18RR.pt
    ```

  * **YAGO3-10**
  
    ```python
    python3 scripts/complex/train.py --dataset YAGO3-10 --optimizer Adagrad --max_epochs 50 --dimension 1000 --batch_size 1000 --learning_rate 0.1 --valid 10 --reg 5e-3
    ```
    ```python
    python3 scripts/complex/test.py --dataset YAGO3-10 --dimension 1000 --learning_rate 0.1 --model_path stored_models/ComplEx_YAGO3-10.pt
    ```
    
* **ConvE**

  * **FB15k**

    ```python
    python3 scripts/conve/train.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 10
    ```
    ```python
    python3 scripts/conve/test.py --dataset FB15k --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --model_path stored_models/ConvE_FB15k.pt
    ```

  * **WN18**
  
    ```python
    python3 scripts/conve/train.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 10
    ```
    ```python
    python3 scripts/conve/test.py --dataset WN18  --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt
    ```
    
  * **FB15k-237**
  
    ```python
    python3 scripts/conve/train.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 10		
    ```
    ```python
    python3 scripts/conve/test.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt
    ```

  * **WN18RR**
  
    ```python
    python3 scripts/conve/train.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 20
    ```
    ```python
    python3 scripts/conve/test.py --dataset WN18RR --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --model_path stored_models/ConvE_WN18RR.pt 
    ```
    
  * **YAGO3-10**
    ```python
    python3 scripts/conve/train.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 10
    ```
    ```python
    python3 scripts/conve/test.py --dataset YAGO3-10 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --model_path stored_models/ConvE_YAGO3-10.pt
    ```

* **TransE**

  * **FB15k**

    ```python
    python3 scripts/transe/train.py  --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2
    ```
    ```python
    python3 scripts/transe/test.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt
    ```

  * **WN18**
  
    ```python
    python scripts/transe/train.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --valid 10
    ```
    ```python
    python scripts/transe/test.py --dataset WN18 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt
    ```

  * **FB15k-237**
  
    ```python
    python3 scripts/transe/train.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --valid 10 --margin 5
    ```
    ```python
    python3 scripts/transe/test.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt
    ```

  * **WN18RR**
  
    ```python
    python scripts/transe/train.py  --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2
    ```
    ```python
    python scripts/transe/test.py  --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt
    ```

  * **YAGO3-10**
  
    ```python
    python3 scripts/transe/train.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50.0 --margin 5 --valid 20
    ```
    ```python
    python3 scripts/transe/test.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50.0 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt
    ```
    

### Extracting and verifying explanations

* **ComplEx**
   * **FB15k**
      * **Necessary Scenario**
         * **Explain predictions:**
            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**
            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode sufficient
           ```
           
  * **WN18**
      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --baseline criage
             ```

         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**
            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient --baseline criage
             ```

         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode sufficient
           ```

  * **FB15k-237**
  
      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**

             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode sufficient
           ```

  * **WN18RR**
  
      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode sufficient
           ```

  * **YAGO3-10**
      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**
 
           * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode sufficient
           ```
    
* **ConvE**

  * **FB15k**
      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_suff.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_suff.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_suff.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --mode sufficient
           ```

  * **WN18**

      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --mode necessary
           ```

    
  * **FB15k-237**
  
      * **Necessary Scenario**

         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python scripts/conve/verify_explanations.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --mode sufficient
           ```

  * **WN18RR**
  
      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --mode sufficient
           ```
    
  * **YAGO3-10**

      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode sufficient --baseline criage
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --mode sufficient
           ```

* **TransE**

  * **FB15k**

      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode necessary --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_input.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode sufficient --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --mode sufficient
           ```


  * **WN18**

      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.01  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode necessary --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 end_to_end_testing/transe/verify_explanations.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.01  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode sufficient --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 end_to_end_testing/transe/verify_explanations.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt --mode sufficient
           ```


  * **FB15k-237**
  
      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode necessary --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --facts_to_explain_path input_facts/transe_fb15k237_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode sufficient --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --mode sufficient
           ```


  * **WN18RR**

      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode necessary --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python scripts/transe/verify_explanations.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode sufficient --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --mode necessary
           ```



  * **YAGO3-10**
  
      * **Necessary Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode necessary
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode necessary --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weigh 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explain predictions:**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode sufficient
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode sufficient --baseline data_poisoning
             ```
             
         * **Verify Explanations:**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weigh 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --mode sufficient
           ```
