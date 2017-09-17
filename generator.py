#-- coding: utf-8 -*-

import os
from six.moves import cPickle
import tensorflow as tf

from kor_char_rnn.model import Model


flags = tf.app.flags
flags.DEFINE_string('word', '삼행시', 'Input korean word (ex. 삼행시)')
FLAGS = flags.FLAGS



class SamhangSiGenerator:

    def __init__(self):
        self.save_data_path = "kor_char_rnn/save"
        self.session = None

    def load_model(self):
        if self.session is None:
            with open(os.path.join(self.save_data_path, 'config.pkl'), 'rb') as f:
                saved_args = cPickle.load(f)
            with open(os.path.join(self.save_data_path, 'chars_vocab.pkl'), 'rb') as f:
                self.chars, self.vocab = cPickle.load(f)

            self.model = Model(saved_args, training=False)

            self.session = tf.Session()
            init = tf.global_variables_initializer()
            self.session.run(init)

            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_data_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.session, ckpt.model_checkpoint_path)

    def generate(self, word):
        result = ""
        for char in word:
            result += self.generate_sentence(char)
        return self.combine_sentence(result, word)

    def combine_sentence(self, result, word):
        result = result.replace("\n", " ")
        for char in word[1:]:
            result = result.replace(char, "\n"+char, 1)
        return result

    def generate_sentence(self, prime):
        sentence_length = 30
        sentences = self.model.sample(self.session, self.chars,
                self.vocab, num=sentence_length, prime=prime)

        return " ".join(sentences.split("\n")[:2])



def main(args):
    samhangsi_generator = SamhangSiGenerator()
    samhangsi_generator.load_model()
    result = samhangsi_generator.generate(FLAGS.word)
    print(result)


if __name__ == '__main__':
    tf.app.run()
