
from hbconfig import Config
import tensorflow as tf



class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def get_train_inputs(X, y):

    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope('training'):

            nonlocal X
            nonlocal y

            X = X.reshape([-1, Config.model.seq_length])
            y = y.reshape([-1, Config.model.seq_length])

            # Define placeholders
            input_placeholder = tf.placeholder(
                tf.int32, X.shape)
            output_placeholder = tf.placeholder(
                tf.int32, y.shape)

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (input_placeholder, output_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(Config.model.batch_size)
            iterator = dataset.make_initializable_iterator()
            next_X, next_y = iterator.get_next()

            tf.identity(next_X[0], 'input_0')
            tf.identity(next_y[0], 'output_0')

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: X,
                               output_placeholder: y})

            # Return batched (features, labels)
            return next_X, next_y

    # Return function and hook
    return train_inputs, iterator_initializer_hook


def get_test_inputs(X, y):

    iterator_initializer_hook = IteratorInitializerHook()

    def test_inputs():
        with tf.name_scope('test'):

            nonlocal X
            nonlocal y

            X = X.reshape([-1, Config.model.seq_length])
            y = y.reshape([-1, Config.model.seq_length])

            # Define placeholders
            input_placeholder = tf.placeholder(
                tf.int32, X.shape)
            output_placeholder = tf.placeholder(
                tf.int32, y.shape)

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (input_placeholder, output_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(Config.model.batch_size)
            iterator = dataset.make_initializable_iterator()
            next_X, next_y = iterator.get_next()

            tf.identity(next_X[0], 'input_0')
            tf.identity(next_y[0], 'output_0')

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: X,
                               output_placeholder: y})

            # Return batched (features, labels)
            return next_X, next_y

    # Return function and hook
    return test_inputs, iterator_initializer_hook

