#-- coding: utf-8 -*-

import argparse
import logging

from hbconfig import Config
import tensorflow as tf

import experiment



def main(mode):
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    run_config = tf.contrib.learn.RunConfig(
            model_dir=Config.train.model_dir)

    if mode == "train":
        experiment.train(run_config, params)
    elif mode == "evaluate":
        experiment.evaluate()
    elif mode == "predict":
        experiment.predict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/evaluate/predict)')
    args = parser.parse_args()

    tf.logging._logger.setLevel(logging.INFO)

    Config(args.config)
    print("Config: ", Config)

    main(args.mode)
