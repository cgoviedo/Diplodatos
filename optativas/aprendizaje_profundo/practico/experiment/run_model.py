import argparse
import sys
import torch.nn as nn

import logging
import mlflow
import torch
import torch.optim as optim

#https://tree.rocks/pytorch-with-multi-process-training-and-get-loss-history-cross-process-running-on-multi-cpu-core-1cbb6df4f5f8

from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .dataset import MeliChallengeDataset
from .utils import PadSequences
from .models import MLPClassifier
from .models import CNNClassifier
from .models import  LSTMClassifier


logging.basicConfig(
    format="%(asctime)s: %(levelname)s - %(message)s",
    level=logging.INFO
)


valid_models = ['MLP' , 'CNN' , 'LSTM']

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

    parser.add_argument("--token-to-index",
                        help="Path to the the json file that maps tokens to indices",
                        required=True)

    parser.add_argument("--pretrained-embeddings",
                        help="Path to the pretrained embeddings file.",
                        required=True)

    parser.add_argument("--language",
                        help="Language working with",
                        required=True)

    parser.add_argument("--test-data",
                        help="If given, use the test data to perform evaluation.")

    parser.add_argument("--validation-data",
                        help="If given, use the validation data to perform evaluation.")


    parser.add_argument("--validation-max-size",
                        help="Maximun size of validation set. Used for try 'small' sets before launch real train",
                        default=None,
                        type=int)

    parser.add_argument("--embeddings-size",
                        default=300,
                        help="Size of the vectors.",
                        type=int)




    parser.add_argument("--random-buffer-size",
                        default=2048,
                        help="Buffer size of trainnig read data (no used for testing nor validation). A buffer of random-buffer-size is read and shuffled. Setting this value to 1 forces the soltion to avoid shuffle",
                        type=int)



    parser.add_argument("--hidden-layers",
                        help="Sizes of the hidden layers of the MLP (can be one or more values)",
                        nargs="+",
                        default=[256, 128],
                        type=int)

    parser.add_argument("--dropout",
                        help="Dropout to apply to each hidden layer",
                        default=0.3,
                        type=float)

    parser.add_argument("--epochs",
                        help="Number of epochs",
                        default=3,
                        type=int)

    parser.add_argument("--freeze-embedings",
                        help="Freeze embedings",
                        default=True,
                        type=bool)

    parser.add_argument("--lr",
                        help="learning rate",
                        default=1e-3,
                        type=float)

    parser.add_argument("--classifier",
                        help="MLP: Multilayer Perceptron, CNN: Convolutional Neural Network",
                        default="MLP"
                        )

    parser.add_argument("--cnn-filters-length",
                    help="CNN Filters size",
                    nargs="+",
                    default=[2,3],
                    type=int)

    parser.add_argument("--cnn-filters-count",
                    help="CNN Filters count",
                    default=200,
                    type=int)



    parser.add_argument("--lstm_hidden_size",
                help="LSTM size of each hidden layer",
                default=64,
                type=int)


    parser.add_argument("--lstm_num_layers",
                help="LSTM number of hidden layers",
                default=10,
                type=int)



    parser.add_argument("--lstm_bias",
                help="LSTM bias",
                default=True,
                type=bool)


    parser.add_argument("--lstm_bidirectional",
                help="LSTM bidirectional",
                default=True,
                type=bool)



    args = parser.parse_args()

    pad_sequences = PadSequences(
        pad_value=0,
        max_length=None,
        min_length=1
    )

    logging.info("Number of Threads: {}".format(torch.get_num_threads()))


    logging.info("Building training dataset")
    train_dataset = MeliChallengeDataset(
        dataset_path=args.train_data,
        random_buffer_size=args.random_buffer_size , # This can be a hypterparameter
        max_size = args.train_max_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # This can be a hyperparameter
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False
    )

    if args.validation_data:
        logging.info("Building validation dataset")
        validation_dataset = MeliChallengeDataset(
            dataset_path=args.validation_data,
            random_buffer_size=1 ,
            max_size = args.validation_max_size
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        validation_dataset = None
        validation_loader = None

    if args.test_data:
        logging.info("Building test dataset")
        test_dataset = MeliChallengeDataset(
            dataset_path=args.test_data,
            random_buffer_size=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None

    mlflow.set_experiment(f"diplodatos.{args.language}")

    with mlflow.start_run():
        logging.info("Starting experiment")

        if args.train_max_size != None:
            logging.info ("Alert! Running experiment with train-max-size {}".format(args.train_max_size))

        if args.validation_max_size != None:
            logging.info ("Alert! Running experiment with validation-max-size {}".format(args.validation_max_size))


        model = None

        if not (args.classifier in valid_models):

            logging.error("Classifier {} does not exists".format(args.classifier))
            sys.exit()

        # Log all relevent hyperparameters
        mlflow.log_params({
        "model_type": args.classifier,
        "embeddings": args.pretrained_embeddings,
        "hidden_layers": args.hidden_layers,
        "dropout": args.dropout,
        "embeddings_size": args.embeddings_size,
        "epochs": args.epochs,
        "freeze_embedings": args.freeze_embedings,
        "lr": args.lr,
        "random-buffer-size": args.random_buffer_size
        })


        if args.classifier == "MLP":

            model = MLPClassifier(
            pretrained_embeddings_path=args.pretrained_embeddings,
            token_to_index=args.token_to_index,
            n_labels=train_dataset.n_labels,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            vector_size=args.embeddings_size,
            freeze_embedings=args.freeze_embedings
            )
        if args.classifier == "CNN":

            model = CNNClassifier(
            pretrained_embeddings_path=args.pretrained_embeddings,
            token_to_index=args.token_to_index,
            n_labels=train_dataset.n_labels,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            vector_size=args.embeddings_size,
            freeze_embedings=args.freeze_embedings,
            filters_length=args.cnn_filters_length,
            filters_count=args.cnn_filters_count
            )


            # Log all relevent hyperparameters
            mlflow.log_params({
            "filters_length":args.cnn_filters_length,
            "filters_count":args.cnn_filters_count
            })



        if args.classifier == "LSTM":

            model = LSTMClassifier(
            pretrained_embeddings_path=args.pretrained_embeddings,
            token_to_index=args.token_to_index,
            vector_size=args.embeddings_size,
            n_labels=train_dataset.n_labels,
            freeze_embedings=args.freeze_embedings,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            dropout=args.dropout,
            bias=True,
            bidirectional=args.lstm_bidirectional
            )





            # Log all relevent hyperparameters
            mlflow.log_params({
            "model_type": model.name,
            "lstm_hidden_size":args.lstm_hidden_size,
            "lstm_num_layers":args.lstm_num_layers
            })


        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        logging.info("Building classifier")



        model = model.to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,  # This can be a hyperparameter
            weight_decay=1e-5  # This can be a hyperparameter
        )

        logging.info("Training classifier")
        for epoch in trange(args.epochs):
            model.train()
            running_loss = []
            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                data = batch["data"].to(device)
                target = batch["target"].to(device)
                output = model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()
                running_loss.append(loss_value.item())
            mlflow.log_metric("train_loss", sum(running_loss) / len(running_loss), epoch)

            if validation_dataset:
                logging.info("Evaluating model on validation")
                model.eval()
                running_loss = []
                targets = []
                predictions = []
                with torch.no_grad():
                    for batch in tqdm(validation_loader):
                        data = batch["data"].to(device)
                        target = batch["target"].to(device)
                        output = model(data)
                        running_loss.append(
                            loss(output, target).item()
                        )
                        targets.extend(batch["target"].numpy())
                        predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                    mlflow.log_metric("validation_loss", sum(running_loss) / len(running_loss), epoch)
                    mlflow.log_metric("validation_bacc", balanced_accuracy_score(targets, predictions), epoch)

        if test_dataset:
            logging.info("Evaluating model on test")
            model.eval()
            running_loss = []
            targets = []
            predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch["data"].to(device)
                    target = batch["target"].to(device)
                    output = model(data)
                    running_loss.append(
                        loss(output, target).item()
                    )
                    targets.extend(batch["target"].numpy())
                    predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                mlflow.log_metric("test_loss", sum(running_loss) / len(running_loss), epoch)
                mlflow.log_metric("test_bacc", balanced_accuracy_score(targets, predictions), epoch)
