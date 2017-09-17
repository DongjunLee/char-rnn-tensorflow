
from hbconfig import Config
import tensorflow as tf

from data_loader import TextLoader
import dataset
from model import CharRNN
import hook



def experiment_fn(run_config, params):

    char_rnn = CharRNN()
    estimator = tf.estimator.Estimator(
            model_fn=char_rnn.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)

    data_loader = TextLoader(Config.data.data_dir, params.batch_size, params.seq_length)
    Config.data.vocab_size = data_loader.vocab_size

    train_input_fn, train_input_hook = dataset.get_train_inputs(data_loader)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=train_input_fn,
        train_steps=Config.train.train_steps,
        train_monitors=[
            train_input_hook,
            hook.print_variables(
                variables=['training/output_0', 'prediction_0'],
                vocab=data_loader.vocab,
                every_n_iter=Config.train.check_hook_n_iter),
            hook.print_variables(
                variables=['loss/reduce_sum'],
                every_n_iter=Config.train.loss_hook_n_iter)],
    )
    return experiment

def train(run_config, params):
    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train",
        hparams=params
    )

def evaluate():
    pass

def predict():
    pass
