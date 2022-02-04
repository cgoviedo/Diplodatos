# Ejecutar los modelos

Antes de correr los modelos, se debe tener creado el env según lo indicado en **0_set_up.ipynb** (este notebook es provisto como parte del material de la materia)


- Ir a la carpeta **practico**

En la carpeta **practico** se debe crear una carpeta o un enlace con el nombre data. En data se debe tener la carpeta **meli-challenge-2019** (provista como parte de material de la materia) y el archivo **SBW-vectors-300-min5.txt.gz** provisto por la materia.

## Desde la carpeta **practico** se puede ejecutar:

   #### MLP

Ejemplo para correr de manera local y ver que el modelo ejecute (se limita a una sola epoc y 50000 samples)


```
python -m experiment.run_model --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz --test-data ./data/meli-challenge-2019/spanish.test.jsonl.gz --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz --language spanish  --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz --embeddings-size 300 --hidden-layers 256 128 --dropout 0.3  --epochs 1 --train-max-size 50000 --validation-max-size 50000 --classifier MLP
```

Ejemplo entrenar con todos los samples

  ```
  python -m experiment.run_model --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz --test-data ./data/meli-challenge-2019/spanish.test.jsonl.gz --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz --language spanish  --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz --embeddings-size 300 --hidden-layers 256 128 --dropout 0.3  --epochs 100 --classifier MLP
  ```

#### CNN


Ejemplo para correr de manera local y ver que el modelo ejecute (se limita a una sola epoch y 50000 samples)  

```
python -m experiment.run_model --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz --test-data ./data/meli-challenge-2019/spanish.test.jsonl.gz --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz --language spanish  --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz --embeddings-size 300 --hidden-layers 256 128 --dropout 0.3  --epochs 1 --train-max-size 50000 --validation-max-size 50000  --random-buffer-size 4096 --cnn-filters-length 3 --classifier CNN
```

Ejemplo entrenar con todos los samples

```
python -m experiment.run_model --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz --test-data ./data/meli-challenge-2019/spanish.test.jsonl.gz --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz --language spanish  --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz --embeddings-size 300 --hidden-layers 256 128 --dropout 0.3  --epochs 100 --random-buffer-size 4096 --cnn-filters-length 3 --classifier CNN
```

   - RNN (LSTM)

Ejemplo para correr de manera local y ver que el modelo ejecute (se limita a una sola epoch y 50000 samples)  

```
python -m experiment.run_model --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz --test-data ./data/meli-challenge-2019/spanish.test.jsonl.gz --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz --language spanish  --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz --embeddings-size 300 --lstm_hidden_size 64 --lstm_num_layers 1  --dropout 0.3  --epochs 1 --train-max-size 500000 --validation-max-size 500000 --classifier LSTM
```

Ejemplo entrenar con todos los samples

```
python -m experiment.run_model --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz --test-data ./data/meli-challenge-2019/spanish.test.jsonl.gz --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz --language spanish  --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz --embeddings-size 300 --lstm_hidden_size 64 --lstm_num_layers 1  --dropout 0.3  --epochs 100 --classifier LSTM
```
