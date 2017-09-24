
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

    data_loader = TextLoader(Config.data.data_dir,
            batch_size=params.batch_size,
            seq_length=params.seq_length)
    Config.data.vocab_size = data_loader.vocab_size

    train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set()

    train_input_fn, train_input_hook = dataset.get_train_inputs(train_X, train_y)
    test_input_fn, test_input_hook = dataset.get_test_inputs(test_X, test_y)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        #min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=[
            train_input_hook,
            hook.print_variables(
                variables=['training/output_0', 'prediction_0'],
                vocab=data_loader.vocab,
                every_n_iter=Config.train.check_hook_n_iter)],
        eval_hooks=[test_input_hook],
        #eval_steps=None
    )
    return experiment
