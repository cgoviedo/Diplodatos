import argparse
import sys
import torch.nn as nn

import logging
import mlflow
import torch
import torch.optim as optim

# https://tree.rocks/pytorch-with-multi-process-training-and-get-loss-history-cross-process-running-on-multi-cpu-core-1cbb6df4f5f8
# https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb#scrollTo=OlXROUWu5Osqç



# https://stackoverflow.com/questions/57248098/using-huggingfaces-pytorch-transformers-gpt-2-for-classifcation-tasks


#  https://jalammar.github.io/illustrated-transformer/

# GPT2
# https://jalammar.github.io/illustrated-gpt2/


# BERT
# https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

# BETO trainning de Bert en español
# https://colab.research.google.com/drive/1uRwg4UmPgYIqGYY4gW_Nsw9782GFJbPt


from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .dataset import MeliChallengeDataset
from .utils import PadSequences
from .models import MLPClassifier
from .models import CNNClassifier
from .models import  LSTMClassifier





if __name__ == "__main__":


    #torch.set_num_threads(200)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data",
                        help="Path to the the training dataset",
                        required=True)

    parser.add_argument("--train-max-size",
                        help="Maximun size of training set. Used for try 'small' sets before launch real train",
                        default=None,
                        type=int)

    args = parser.parse_args()



    train_dataset = MeliChallengeDataset(
        dataset_path=args.train_data,
        random_buffer_size=2048 , # This can be a hypterparameter
        max_size = args.train_max_size
    )


    for item in train_dataset:
        print (item)
        break;
