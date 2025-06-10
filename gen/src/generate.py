import argparse
import math
import numpy as np
import os
from pathlib import Path
import pickle
import sys
import tensorflow as tf
import time
import uuid


def load_generator(model):
	with open(model, 'rb') as f:
		bytes = f.read()
	training_state: TrainingState = pickle.loads(bytes)
	return training_state.visualization_generator


def generate(generator, vectors, dir_out, batch_size=16):
	prog_bar = tf.keras.utils.Progbar(len(vectors) // batch_size)
	vectors = tf.convert_to_tensor(vectors)
	pad = int(math.log10(len(vectors) - 1) + 1)
	for start in range(0, len(vectors), batch_size): # vectors.shape[0] ---> len(vectors)
		images = []
		end = start + batch_size
		batch = vectors[start:end]
		image_batch = generator(batch)
		images.append(image_batch)
		prog_bar.add(1)
		images = tf.concat(images, axis=0)
		for i, image in enumerate(images):
		    img_no = start + i
		    file_path = os.path.join(dir_out, f'{img_no:0{pad}}.png')
		    image = tf.convert_to_tensor(image)
		    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
		    image = tf.io.encode_png(image).numpy()
		    with open(file_path, 'wb') as f:
		        f.write(image)





















