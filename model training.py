
import os 
import logging
import warnings
warnings.filterwarnings("ignore")
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
random.seed(42)
np.random.seed(42)
os.chdir('D:\dy\prj')



dt = np.load('user_data.npy',allow_pickle=True)
dt = pad_dt(dt)

test_sample = dt[:1000]

s = time.time()
tf.ragged.constant(test_sample[:10],ragged_rank=1)
e = time.time()
print(f'{(e-s)*100}s estimated')

test_sample = tf.ragged.constant(test_sample,ragged_rank=1)

tvae = TVAE(    num_layers=12,
    d_model=16,
    num_heads=8,
    dff=512,
    input_vocab_size=66,
    target_vocab_size=66,
    dropout_rate=0.1)

tvae.train_and_save(test_sample)






