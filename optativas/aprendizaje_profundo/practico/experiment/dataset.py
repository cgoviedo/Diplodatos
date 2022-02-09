import gzip
import json
import random

from torch.utils.data import IterableDataset


from transformers import BertTokenizer






class BaseItemDecorator:

    def decorate(self, item):

        item = {
            "data": item["data"],
            "target": item["target"]
        }

        return item


class BertItemDecorator:

    def __init__(self):
            self.tokenizer = BertTokenizer.from_pretrained("pytorch/", do_lower_case=False)

    def decorate(self, item):

        item = {
            "data": self.tokenizer(item["title"],padding='max_length', max_length = 50, truncation=True, return_tensors="pt"),
            "target": item["target"]
        }

        return item


defaultItemDecorator = BaseItemDecorator()
bertItemDecorator = BertItemDecorator()

class MeliChallengeDataset(IterableDataset):
    def __init__(self,
                 dataset_path,
                 random_buffer_size=2048,
                 max_size=None , tokenizer = None , itemDecorator = defaultItemDecorator):

        assert random_buffer_size > 0
        self.dataset_path = dataset_path
        self.random_buffer_size = random_buffer_size
        self.max_size = max_size
        self.itemDecorator = itemDecorator

        with gzip.open(self.dataset_path, "rt") as dataset:
            item = json.loads(next(dataset).strip())
            self.n_labels = item["n_labels"]
            print ("Cantidad de labels: {}".format(self.n_labels))
            self.dataset_size = item["size"]

            if (self.max_size != None):
                self.dataset_size = self.max_size if self.max_size < self.dataset_size else self.dataset_size

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        try:
            with gzip.open(self.dataset_path, "rt") as dataset:
                shuffle_buffer = []

                for index, line in enumerate(dataset):

                    if index + 1 == self.dataset_size:
                        break

                    item = json.loads(line.strip())

                    item = self.itemDecorator.decorate(item)


                    if self.random_buffer_size == 1:
                        yield item

                    else:
                        shuffle_buffer.append(item)

                        if len(shuffle_buffer) == self.random_buffer_size:
                            random.shuffle(shuffle_buffer)
                            for item in shuffle_buffer:
                                yield item
                            shuffle_buffer = []

                if len(shuffle_buffer) > 0:
                    random.shuffle(shuffle_buffer)
                    for item in shuffle_buffer:
                        yield item
        except GeneratorExit:
            return
