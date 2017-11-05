#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys
import os
from model import speaker_recognition
from train_iter import *
import imp
import pdb

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "base save path for the experiment")
flags.DEFINE_string("eval_only", "False",
                    "Only Evaluate.No Training.")
flags.DEFINE_string("log_path",None,"Log Directory path")

FLAGS = flags.FLAGS

def main(_):
    model_config = imp.load_source('config', 'config/config.py').config().speaker_reco

    print("Config :")
    print(model_config)
    print("\n")

    save_path = os.path.join(FLAGS.save_path, model_config.save_path)
    log_path = os.path.join(FLAGS.log_path,'log_folder')

    with tf.Graph().as_default():
        main_model = eval(model_config.model)(model_config)

        if model_config.load_mode == "continue":
            if not tf.gfile.Exists(save_path):
                os.makedirs(save_path)
                os.chmod(save_path, 0775)
        model_vars = main_model.model_vars()

        sv = tf.train.Supervisor( logdir=log_path, init_feed_dict=main_model.init_feed_dict())
        
        with sv.managed_session() as session:
            if model_config.load_mode == "best":
                sv.saver.restore(
                    sess=session,
                    save_path=os.path.join(save_path, "best_model.ckpt"))


            i, patience = 0, 0
            best_valid_metric = 1e10

            while patience < model_config.patience:
                i += 1

                iterator_train = SSIterator(model_config.batch_size,model_config,1234,'train')
                iterator_valid = SSIterator(model_config.batch_size,model_config,1234,'valid')
                    # Initialise the iterator for dev set and train set
                print("\nEpoch: %d" % (i))
                main_model.run_epoch(session, reader = iterator_train, is_training=True, verbose=True)

                valid_loss = main_model.run_epoch(session, reader = iterator_valid, verbose=True)

                main_model.find_accuracy(session,reader = iterator_valid,verbose=True)

                if best_valid_metric > valid_loss:
                    best_valid_metric = valid_loss

                    print("\nsaving best model...")
                    sv.saver.save(sess=session, save_path=os.path.join(save_path, "best_model.ckpt"))
                    patience = 0
                else:
                    patience += 1
                    print("\nLosing patience...")


            if FLAGS.save_path and model_config.load_mode == "fresh":
                print("\nSaving model to %s." % save_path)
                sv.saver.save(session, save_path, global_step=sv.global_step)

if __name__ == "__main__":
    tf.app.run()
