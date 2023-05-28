class KelpieExplanation:
    _original_model_results = {}  # map original samples to scores and ranks from the original model
    _base_pt_model_results = {}   # map original samples to scores and ranks from the base post-trained model

    # The kelpie_cache is a simple LRU cache that allows reuse of KelpieDatasets and of base post-training results
    # without need to re-build them from scratch every time.
    _kelpie_dataset_cache_size = 20
    _kelpie_dataset_cache = OrderedDict()

    """
        Given a "sample to explain" (that is, a sample that the model currently predicts as true,
        and that we want to be predicted as false);
        and given and a list of training samples containing the entity to convert;
        compute the relevance of the samples in removal, that is, an estimate of the effect they would have
        if removed (all together) from the perspective entity to worsen the prediction of the sample to convert.

        :param sample_to_explain: the sample that we would like the model to predict as "true",
                                    in the form of a tuple (head, relation, tail)
        :param samples_to_remove:   the list of samples containing the perspective entity
                                    that we want to analyze the effect of, if added to the perspective entity
    """
    df = pd.DataFrame(columns=['sample_to_explain', 'samples_to_remove', 'length', 'base_score', 'base_best', 'base_rank', 'pt_score', 'pt_best', 'pt_rank', 'rank_worsening', 'score_worsening', 'relevance'])

    def __init__(self, 
                 sample_to_explain: Tuple[Any, Any, Any],
                 samples_to_remove: list) -> None:
        logger.info("Create kelpie explanation on sample: %s", dataset.sample_to_fact(sample_to_explain, True))
        print("Removing sample:", [dataset.sample_to_fact(x, True) for x in samples_to_remove])
        self.sample_to_explain = sample_to_explain
        self.samples_to_remove = samples_to_remove
        
        self.head = sample_to_explain[0]
        self.kelpie_dataset = self._get_kelpie_dataset_for(original_entity_id=self.head)
        self.kelpie_sample_to_predict = self.kelpie_dataset.as_kelpie_sample(original_sample=self.sample_to_explain)

        kelpie_init_tensor_size = model.dimension if not isinstance(model, TuckER) else model.entity_dimension
        self.kelpie_init_tensor = torch.rand(1, kelpie_init_tensor_size, device='cuda') - 0.5

        base_pt_target_entity_score, \
        base_pt_best_entity_score, \
        base_pt_target_entity_rank = self.post_training_results_multiple()

        pt_target_entity_score, \
        pt_best_entity_score, \
        pt_target_entity_rank = self.post_training_results_multiple(self.samples_to_remove)

        rank_worsening = (pt_target_entity_rank - base_pt_target_entity_rank) / base_pt_target_entity_rank
        score_worsening = base_pt_target_entity_score - pt_target_entity_score
        if model.is_minimizer():
            score_worsening *= -1

        # logger.info(f"Kelpie explanation created. Rank worsening: {rank_worsening}, score worsening: {score_worsening}")

        self.relevance = get_removel_relevance(rank_worsening, score_worsening)
        self.ret = {'sample_to_explain': dataset.sample_to_fact(sample_to_explain, True),
                'samples_to_remove': [dataset.sample_to_fact(x, True) for x in samples_to_remove],
                'length': len(samples_to_remove),
                'base_score': base_pt_target_entity_score,
                'base_best': base_pt_best_entity_score,
                'base_rank': base_pt_target_entity_rank,
                'pt_score': pt_target_entity_score,
                'pt_best': pt_best_entity_score,
                'pt_rank': pt_target_entity_rank,
                'rank_worsening': rank_worsening,
                'score_worsening': score_worsening,
                'relevance': self.relevance}
        
        self.df.loc[len(self.df)] = self.ret
        self.df.to_csv(os.path.join(args.output_folder, f"output_details.csv"), index=False)
        logger.info(f"Kelpie explanation created. {str(self.ret)}")


    def post_training_results_multiple(self, samples_to_remove: list = []):
        if len(samples_to_remove) == 0 and self.sample_to_explain in self._base_pt_model_results:
            return self._base_pt_model_results[self.sample_to_explain]
        results = []
        logger.info(f'[post_training_results_multiple] {len(self.kelpie_dataset.kelpie_train_samples)} - {len(samples_to_remove)}, {post_train_times} times x {hyperparameters[RETRAIN_EPOCHS]} epoches')
        for _ in tqdm(range(post_train_times)):
            # results.append(self.post_training_results(samples_to_remove))
            results.append(self.post_training_save(samples_to_remove))
        target_entity_score, \
        best_entity_score, \
        target_entity_rank = zip(*results)
        if len(samples_to_remove) == 0:
            self._base_pt_model_results[self.sample_to_explain] = mean(target_entity_score), mean(best_entity_score), mean(target_entity_rank)
        logger.info(f'score: {mean(target_entity_score)} ± {std(target_entity_score)}, best: {mean(best_entity_score)} ± {std(best_entity_score)}, rank: {mean(target_entity_rank)} ± {std(target_entity_rank)}')
        return mean(target_entity_score), mean(best_entity_score), mean(target_entity_rank)

    def _get_kelpie_dataset_for(self, original_entity_id: int) -> KelpieDataset:
        """
        Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """

        if original_entity_id not in self._kelpie_dataset_cache:
            kelpie_dataset = KelpieDataset(dataset=dataset, entity_id=original_entity_id)
            self._kelpie_dataset_cache[original_entity_id] = kelpie_dataset
            self._kelpie_dataset_cache.move_to_end(original_entity_id)

            if len(self._kelpie_dataset_cache) > self._kelpie_dataset_cache_size:
                self._kelpie_dataset_cache.popitem(last=False)

        return self._kelpie_dataset_cache[original_entity_id]

    def original_results(self) :
        sample = self.sample_to_explain
        if not sample in self._original_model_results:
            target_entity_score, \
            best_entity_score, \
            target_entity_rank = extract_detailed_performances(model, sample)
            self._original_model_results[sample] = (target_entity_score, best_entity_score, target_entity_rank)
        return self._original_model_results[sample]
    
    def post_training_clone(self, samples_to_remove: numpy.array=[]):
        print('origin', extract_detailed_performances(model, self.sample_to_explain))
        
        post_model = PostConvE(model, self.head, self.kelpie_init_tensor)
        post_model = post_model.to('cuda')
        # Now you can do your training. Only the entity_embeddings for self.head will get updated...
        # optimizer = kelpie_optimizer_class(model=post_model,
        #                                     hyperparameters=hyperparameters,
        #                                     verbose=False)
        optimizer = Optimizer(model=post_model, hyperparameters=hyperparameters, verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]

        original_train_samples = extract_samples_with_entity(dataset.train_samples, self.head)
        ids = []
        for sample in samples_to_remove:
            ids.append(original_train_samples.tolist().index(list(sample)))
        post_train_samples = np.delete(original_train_samples, ids, axis=0)

        print('original, post_train = ', len(original_train_samples), len(post_train_samples))
        # post_train_samples = torch.tensor(post_train_samples).to('cuda')
        optimizer.train(train_samples=post_train_samples)
        ret = extract_detailed_performances(post_model, self.sample_to_explain)
        print('pt', ret)

        return ret
    
    def post_training_directly(self, samples_to_remove: numpy.array=[]):
        print('origin', extract_detailed_performances(model, self.sample_to_explain))

        # for param in model.parameters():
        #     param.requires_grad = False
        
        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        entity_embeddings = frozen_entity_embeddings.clone()
        trainable_head_embedding = torch.nn.Parameter(self.kelpie_init_tensor, requires_grad=True)
        entity_embeddings[self.head] = trainable_head_embedding
        model.entity_embeddings = torch.nn.Parameter(entity_embeddings)

        # Now you can do your training. Only the entity_embeddings for self.head will get updated...
        optimizer = Optimizer(model=model, hyperparameters=hyperparameters, verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]

        original_train_samples = extract_samples_with_entity(dataset.train_samples, self.head)
        ids = []
        for sample in samples_to_remove:
            ids.append(original_train_samples.tolist().index(list(sample)))
        post_train_samples = np.delete(original_train_samples, ids, axis=0)

        print('original, post_train = ', len(original_train_samples), len(post_train_samples))
        optimizer.train(train_samples=post_train_samples)
        ret = extract_detailed_performances(model, self.sample_to_explain)
        print('pt', ret)

        model.entity_embeddings = torch.nn.Parameter(frozen_entity_embeddings)

        return ret
    
    def post_training_save(self, samples_to_remove: numpy.array=[]):
        model = TargetModel(dataset=dataset, hyperparameters=hyperparameters)
        model = model.to('cuda')
        model.load_state_dict(state_dict=args.state_dict)

        # print('origin', extract_detailed_performances(model, self.sample_to_explain))
        for param in model.parameters():
            if param.is_leaf:
                param.requires_grad = False
        
        # frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        # trainable_head_embedding = torch.nn.Parameter(self.kelpie_init_tensor, requires_grad=True)
        # frozen_entity_embeddings[self.head] = trainable_head_embedding
        # print(type(frozen_entity_embeddings))
        # model.entity_embeddings.requires_grad = True
        # model.frozen_indices = [i for i in range(model.entity_embeddings.shape[0]) if i != self.head]
        model.start_post_train(trainable_indices=[self.head], init_tensor=self.kelpie_init_tensor)

        # print('weight', tensor_head(model.convolutional_layer.weight))
        # print('embedding', tensor_head(model.entity_embeddings[self.head]))
        # print('other embedding', tensor_head(model.entity_embeddings[self.head+1]))

        # Now you can do your training. Only the entity_embeddings for self.head will get updated...
        optimizer = Optimizer(model=model, hyperparameters=hyperparameters, verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]
        # optimizer.learning_rate *= 10

        original_train_samples = extract_samples_with_entity(dataset.train_samples, self.head)
        ids = []
        for sample in samples_to_remove:
            ids.append(original_train_samples.tolist().index(list(sample)))
        post_train_samples = np.delete(original_train_samples, ids, axis=0)

        # print('original, post_train = ', len(original_train_samples), len(post_train_samples))
        optimizer.train(train_samples=post_train_samples, post_train=True)
        ret = extract_detailed_performances(model, self.sample_to_explain)
        # print('pt', ret)

        # print('weight', tensor_head(model.convolutional_layer.weight))
        # print('embedding', tensor_head(model.entity_embeddings[self.head]))
        # print('other embedding', tensor_head(model.entity_embeddings[self.head+1]))
        
        return ret


    def post_training_results(self, samples_to_remove: numpy.array=[]):
        print('origin', extract_detailed_performances(model,self.sample_to_explain))
        kelpie_model = kelpie_model_class(model=model,
                                        dataset=self.kelpie_dataset,
                                        init_tensor=self.kelpie_init_tensor)
        print('base origin', extract_detailed_performances(kelpie_model,self.sample_to_explain))
        print('base kelpie', extract_detailed_performances(kelpie_model,self.kelpie_sample_to_predict))
        if len(samples_to_remove):
            self.kelpie_dataset.remove_training_samples(samples_to_remove)
        base_pt_model = self.post_train(kelpie_model_to_post_train=kelpie_model,
                                        kelpie_train_samples=self.kelpie_dataset.kelpie_train_samples) # type: KelpieModel
        print('pt origin', extract_detailed_performances(kelpie_model,self.sample_to_explain))
        print('pt kelpie', extract_detailed_performances(kelpie_model,self.kelpie_sample_to_predict))
        if len(samples_to_remove):
            self.kelpie_dataset.undo_last_training_samples_removal()
        return extract_detailed_performances(base_pt_model, self.kelpie_sample_to_predict)

    def post_train(self,
                   kelpie_model_to_post_train: KelpieModel,
                   kelpie_train_samples: numpy.array):
        """
        :param kelpie_model_to_post_train: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_train_samples:
        :return:
        """
        kelpie_model_to_post_train.to('cuda')
        optimizer = kelpie_optimizer_class(model=kelpie_model_to_post_train,
                                            hyperparameters=hyperparameters,
                                            verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]
        # print(optimizer.epochs)
        t = time.time()
        optimizer.train(train_samples=kelpie_train_samples)
        # print(f'[post_train] kelpie_train_samples: {len(kelpie_train_samples)}, epoches: {optimizer.epochs}, time: {rd(time.time() - t)}')
        return kelpie_model_to_post_train
    