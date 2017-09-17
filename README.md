# char-rnn

This code implements multi-layer Recurrent Neural Network (RNN, LSTM, and GRU) for training/sampling from character-level language models.

## Requirements

- Python 3.6
- TensorFlow >= 1.3
- hb-config

## Features

- Using Higher-APIs in TensorFlow
	- [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
	- [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment)
	- [Dataset](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset)
- Can handle **Any Language**
- Korean SamhangSi (like acrostic poem)

## Usage

First, need to check this model working. 

```bash
python main.py --config check_tiny_overfitting --mode train
```

## Todo

- implemens evaluate, predict
- generator.py for samhangsi


## Reference

- [sherjilozair/char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow)
- [insikk/kor-char-rnn-tensorflow](https://github.com/insikk/kor-char-rnn-tensorflow)
- [Higher-Level APIs in TensorFlow](https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0)