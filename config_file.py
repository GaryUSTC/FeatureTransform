import os
import numpy as np


def initialize_GPU():
    # Initialize GPUs
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session
