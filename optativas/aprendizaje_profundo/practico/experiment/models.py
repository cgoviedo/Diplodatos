
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel
from transformers import BertForSequenceClassification
from transformers import BertForPreTraining

import gzip
import json





class BaseClassifier(nn.Module):

    def __init__(self,
                 device
                 ):
        super().__init__()
        self.device = device


    def forward_pass(self, input):
        data = input["data"].to(self.device)
        output = self.__call__(data)
        return output

# https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

class BertClassifier(BaseClassifier):


    def __init__(self, device, dropout=0.5):

        super(BertClassifier, self).__init__(device)

        #self.bert = BertModel.from_pretrained("pytorch/")
        #self.bert = BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        self.bert = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        #self.bert = BertForPreTraining.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")



        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 632)
        self.relu = nn.ReLU()

    def forward_pass(self, input):

        #print (input)
        #print (type(input['data']))
        mask = input['data']['attention_mask'].to(self.device)
        input_id = input['data']['input_ids'].squeeze(1).to(self.device)
        output = self.__call__(input_id,mask)
        #return output.argmax(dim=1)
        return output


    def forward(self, input_id, mask):


        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class EmbeddingsBaseClassifier(BaseClassifier):
        def __init__(self,
                     device,
                     pretrained_embeddings_path,
                     token_to_index,
                     n_labels,
                     dropout,
                     vector_size=300,
                     freeze_embedings=True

                     ):
            super().__init__(device)
            with gzip.open(token_to_index, "rt") as fh:
                token_to_index = json.load(fh)
            embeddings_matrix = torch.randn(len(token_to_index), vector_size)
            embeddings_matrix[0] = torch.zeros(vector_size)
            with gzip.open(pretrained_embeddings_path, "rt") as fh:
                next(fh)
                for line in fh:
                    word, vector = line.strip().split(None, 1)
                    if word in token_to_index:
                        embeddings_matrix[token_to_index[word]] =\
                            torch.FloatTensor([float(n) for n in vector.split()])
            self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                           freeze=freeze_embedings,
                                                           padding_idx=0)

            self.vector_size = vector_size

            self.dropout = dropout
            self.n_labels = n_labels

            print (" BaseClassifier :Cantidad de labels {} ".format(self.n_labels))



class MLPClassifier(EmbeddingsBaseClassifier):
    def __init__(self,
                 device,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 hidden_layers=[256, 128],
                 dropout=0.3,
                 vector_size=300,
                 freeze_embedings=True
                 ):

        super().__init__(
                     device,
                     pretrained_embeddings_path,
                     token_to_index,
                     n_labels,
                     dropout,
                     vector_size,
                     freeze_embedings
                     )

        self.hidden_layers = [
            nn.Linear(vector_size, hidden_layers[0])
        ]
        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(
                nn.Linear(input_size, output_size)
            )

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(hidden_layers[-1], self.n_labels)
        self.name = "Perceptron Multicapa"

    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x, dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)
        x = self.output(x)
        #print ("{} - {}".format(x, len(x[0])))

        return x


class MLPBagClassifier(EmbeddingsBaseClassifier):
    def __init__(self,
                 device,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 hidden_layers=[256, 128],
                 dropout=0.3,
                 vector_size=300,
                 freeze_embedings=True
                 ):

        super().__init__(
                     device,
                     pretrained_embeddings_path,
                     token_to_index,
                     n_labels,
                     dropout,
                     vector_size,
                     freeze_embedings
                     )


        self.embedding_bag = nn.EmbeddingBag.from_pretrained(self.embeddings.weight, padding_idx=self.embeddings.padding_idx, mode='sum')

        self.hidden_layers = [
            nn.Linear(vector_size, hidden_layers[0])
        ]
        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(
                nn.Linear(input_size, output_size)
            )

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(hidden_layers[-1], self.n_labels)
        self.name = "Perceptron Multicapa Bag"

    def forward(self, x):
        x = self.embedding_bag(x)
        #x = torch.mean(x, dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)
        x = self.output(x)
        #print ("{} - {}".format(x, len(x[0])))

        return x

    def init_weights(self):
        initrange = 0.5
        self.embedding_bag.weight.data.uniform_(-initrange, initrange)
        #self.fc.weight.data.uniform_(-initrange, initrange)
        #self.fc.bias.data.zero_()



class CNNClassifier(EmbeddingsBaseClassifier):

    def __init__(self,
                 device,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 hidden_layers=[256, 128],
                 dropout=0.3,
                 vector_size=50,
                 freeze_embedings=True,
                 filters_length=[5],
                 filters_count=100
                 ):

        super().__init__(
                 device,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 dropout,
                 vector_size,
                 freeze_embedings
                 )


        self.convs = []

        self.hidden_layers = []

        for filter_length in filters_length:
            self.convs.append(
                nn.Conv1d(vector_size, filters_count, filter_length)
            )
        self.convs = nn.ModuleList(self.convs)

        #self.fc = nn.Linear(filters_count * len(filters_length), 128)

        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(nn.Linear(input_size, output_size))

        self.fc = nn.Linear(filters_count * len(filters_length), hidden_layers[0])



        self.output = nn.Linear(hidden_layers[-1], n_labels)

        self.name = "CNN"

    @staticmethod
    def conv_global_max_pool(x, conv):
        return F.relu(conv(x).transpose(1, 2).max(1)[0])

    def forward(self, x):
        x = self.embeddings(x).transpose(1, 2)  # Conv1d takes (batch, channel, seq_len)
        x = [self.conv_global_max_pool(x, conv) for conv in self.convs]

        x = torch.cat(x, dim=1)
        x = F.relu(self.fc(x))
        #if self.dropout:
        #    x = F.dropout(x, self.dropout)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)

        #x = torch.sigmoid(self.output(x))
        x = self.output(x)

        return x



class LSTMClassifier(EmbeddingsBaseClassifier):
    def __init__(self,
                 device,
                 pretrained_embeddings_path,
                 token_to_index,
                 vector_size,
                 n_labels ,
                 freeze_embedings=True,
                 lstm_hidden_size=32,
                 lstm_num_layers=1,
                 dropout=0.,
                 bias=True,
                 bidirectional=False
                 ):

        super().__init__(
                 device,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 dropout,
                 vector_size,
                 freeze_embedings
                 )

        self.name = "LSTM"


        # Set our LSTM parameters
        self.lstm_config = {'input_size': vector_size,
                            'hidden_size': lstm_hidden_size,
                            'num_layers': lstm_num_layers,
                            'bias': bias,
                            'batch_first': True,
                            'dropout': dropout,
                            'bidirectional': True}

        # Set our fully connected layer parameters
        self.linear_config = {'in_features': lstm_hidden_size * 2 if bidirectional else lstm_hidden_size ,
                              'out_features': n_labels,
                              'bias': bias}

        # Instanciate the layers
        self.lstm = nn.LSTM(**self.lstm_config)
        self.classification_layer = nn.Linear(**self.linear_config)
        #self.activation = nn.Sigmoid()

    def forward(self, inputs):
        emb = self.embeddings(inputs)
        #print(emb.shape)
        lstm_out, _ = self.lstm(emb)
        #print(lstm_out.shape)
        # Take last state of lstm, which is a representation of
        # the entire text
        lstm_out = lstm_out[:, -1, :].squeeze()
        #print(lstm_out.shape)
        #predictions = self.activation(self.classification_layer(lstm_out))

        predictions = self.classification_layer(lstm_out)

        # print(prediction.shape)
        return predictions
