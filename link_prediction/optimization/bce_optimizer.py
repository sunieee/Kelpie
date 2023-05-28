import tqdm
import torch
import numpy as np
from torch import optim
from collections import defaultdict
import os

from link_prediction.models.conve import ConvE
from link_prediction.models.model import Model, BATCH_SIZE, LABEL_SMOOTHING, LEARNING_RATE, DECAY, EPOCHS
from link_prediction.optimization.optimizer import Optimizer


def rd(x):
    return round(x, 4)

def tensor_head(t):
    return [rd(x) for x in t.view(-1)[:3].detach().cpu().numpy().tolist()]


class BCEOptimizer(Optimizer):
    """
        This optimizer relies on BCE loss.
        Instead of considering each training sample as the "unit" for training,
        it groups training samples into couples (h, r) -> [all t for which <h, r, t> in training set].
        Each couple (h, r) with the corresponding tails is treated as if it was one sample.

        When passing them to the loss...

        In our implementation, it is used by the following models:
            - TuckER
            - ConvE

    """

    def __init__(self,
                 model: Model,
                 hyperparameters: dict,
                 verbose: bool = True):
        """
            BCEOptimizer initializer.
            :param model: the model to train
            :param hyperparameters: a dict with the optimization hyperparameters. It must contain at least:
                    - BATCH SIZE
                    - LEARNING RATE
                    - DECAY
                    - LABEL SMOOTHING
                    - EPOCHS
            :param verbose:
        """

        Optimizer.__init__(self, model=model, hyperparameters=hyperparameters, verbose=verbose)

        self.args = model.args
        self.train_restrain = self.args.train_restrain
        self.tail_restrain = self.args.tail_restrain
        self.batch_size = hyperparameters[BATCH_SIZE]
        self.label_smoothing = hyperparameters[LABEL_SMOOTHING]
        self.learning_rate = hyperparameters[LEARNING_RATE]
        self.decay = hyperparameters[DECAY]
        self.epochs = hyperparameters[EPOCHS]

        self.loss = torch.nn.BCELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)  # we only support ADAM for BCE
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.decay)

    def train(self,
              train_samples: np.array,
              save_path: str = None,
              evaluate_every:int =-1,
              valid_samples:np.array = None,
              post_train: bool = False):

        all_training_samples = np.vstack((train_samples, self.dataset.invert_samples(train_samples)))
        er_vocab = self.extract_er_vocab(all_training_samples)
        er_vocab_pairs = list(er_vocab.keys())
        er_vocab_pairs.sort(key=lambda x: x[1]) # 对样例按照relation排序！

        self.model.cuda()

        for e in range(1, self.epochs+1):
            self.epoch(er_vocab=er_vocab, er_vocab_pairs=er_vocab_pairs, batch_size=self.batch_size, post_train=post_train)

            if evaluate_every > 0 and valid_samples is not None and e % evaluate_every == 0:
                self.model.eval()
                self.evaluator.evaluate(samples=valid_samples, write_output=False)

                if save_path is not None:
                    print("saving model...")
                    torch.save(self.model.state_dict(), save_path)
                print("done.")

        if save_path is not None:
            print("saving model...")
            torch.save(self.model.state_dict(), save_path)
            print("done.")

    def extract_er_vocab(self, samples):
        er_vocab = defaultdict(list)
        for sample in samples:
            er_vocab[(sample[0], sample[1])].append(sample[2])
        return er_vocab

    def extract_batch(self, er_vocab, er_vocab_pairs, batch_start, batch_size):
        batch = er_vocab_pairs[batch_start: min(batch_start+batch_size, len(er_vocab_pairs))]

        targets = np.zeros((len(batch), self.dataset.num_entities))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        if self.label_smoothing:
            targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.shape[1])

        batch = np.array(batch)
        return torch.tensor(batch).cuda(), torch.FloatTensor(targets).cuda()

    def epoch(self,
              er_vocab,
              er_vocab_pairs,
              batch_size: int,
              post_train: bool = False):

        if self.tail_restrain is None:
            np.random.shuffle(er_vocab_pairs)
        self.model.train()

        with tqdm.tqdm(total=len(er_vocab_pairs), unit='ex', disable=not self.verbose) as bar:
            bar.set_description('train loss')
            batch_start = 0

            while batch_start < len(er_vocab_pairs):
                batch, targets = self.extract_batch(er_vocab=er_vocab,
                                                    er_vocab_pairs=er_vocab_pairs,
                                                    batch_start=batch_start,
                                                    batch_size=batch_size)
                l = self.step_on_batch(batch, targets, post_train=post_train)

                if post_train:
                    # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES
                    # print(tensor_head(self.model.trainable_entity_embeddings))
                    self.model.update_embeddings()

                batch_start+=batch_size
                bar.update(batch_size)
                bar.set_postfix(loss=str(round(l.item(), 6)))

            if self.decay and self.scheduler.optimizer._step_count:
                self.scheduler.step()


    def step_on_batch(self, batch, targets, post_train: bool = False):
        if isinstance(self.model, ConvE):
            if len(batch) == 1:
                # if the batch has length 1 ( = this is the last batch) and the model has batch_norm layers,
                # do not try to update the batch_norm layers, because they would not work.
                self.model.batch_norm_1.eval()
                self.model.batch_norm_2.eval()
                self.model.batch_norm_3.eval()
            if post_train:
                # just making sure that these layers are still in eval() mode
                self.model.batch_norm_1.eval()
                self.model.batch_norm_2.eval()
                self.model.batch_norm_3.eval()
                self.model.convolutional_layer.eval()
                self.model.hidden_layer.eval()

        self.optimizer.zero_grad()
        if self.train_restrain:
            predictions = self.model.forward(batch, restrain=True)
            if self.tail_restrain:
                targets = targets[:, self.model.get_tail_set(batch, '-')]
        else:
            predictions = self.model.forward(batch)
        loss = self.loss(predictions, targets)
        # 发现tail_size为0，使得按照BCE公式分母为0 => loss=nan
        if np.isnan(loss.item()):
            print(predictions.size(), targets.size())
            print(batch.tolist()[0])
            os.abort()
        loss.backward()
        self.optimizer.step()

        # if the layers had been set to mode "eval", put them back to mode "train"
        if len(batch) == 1 and isinstance(self.model, ConvE):
            self.model.batch_norm_1.train()
            self.model.batch_norm_2.train()
            self.model.batch_norm_3.train()

        return loss

class KelpieBCEOptimizer(BCEOptimizer):
    def __init__(self,
                 model:Model,
                 hyperparameters: dict,
                 verbose: bool = True):

        super(KelpieBCEOptimizer, self).__init__(model=model,
                                                 hyperparameters=hyperparameters,
                                                 verbose=verbose)

        self.optimizer = optim.Adam(params=self.model.parameters())

    # Override
    def epoch(self,
              er_vocab,
              er_vocab_pairs,
              batch_size: int):

        # np.random.shuffle(er_vocab_pairs)
        self.model.train()

        with tqdm.tqdm(total=len(er_vocab_pairs), unit='ex', disable=not self.verbose) as bar:
            bar.set_description('train loss')

            batch_start = 0
            while batch_start < len(er_vocab_pairs):
                batch, targets = self.extract_batch(er_vocab=er_vocab,
                                                    er_vocab_pairs=er_vocab_pairs,
                                                    batch_start=batch_start,
                                                    batch_size=batch_size)
                l = self.step_on_batch(batch, targets)

                # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
                # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
                # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
                self.model.update_embeddings()

                batch_start+=batch_size
                bar.update(batch_size)
                bar.set_postfix(loss=str(round(l.item(), 6)))

            if self.decay and self.scheduler.optimizer._step_count:
                self.scheduler.step()


    def step_on_batch(self, batch, targets):
        # batch = batch.cuda()
        # targets = targets.cuda()

        # just making sure that these layers are still in eval() mode
        if isinstance(self.model, ConvE):
            self.model.batch_norm_1.eval()
            self.model.batch_norm_2.eval()
            self.model.batch_norm_3.eval()
            self.model.convolutional_layer.eval()
            self.model.hidden_layer.eval()

        self.optimizer.zero_grad()
        predictions = self.model.forward(batch)

        print('[step]batch:', batch.shape)
        print('[step]predictions:', predictions.shape)
        print('[step]targets:', targets.shape)

        loss = self.loss(predictions, targets)
        loss.backward()
        self.optimizer.step()

        return loss
