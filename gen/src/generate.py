import argparse
import numpy as np
import os
from pathlib import Path
import pickle
import sys
import tensorflow as tf
import time
from train import TrainingState
import uuid


def load_generator(model, path='model'):
	with open(f'{path}/{model}', 'rb') as f:
		bytes = f.read()
	training_state: TrainingState = pickle.loads(bytes)
	return training_state.visualization_generator


def generate(model, vectors, dir_out, batch_size=32):
	generator = load_generator(model)
	prog_bar = tf.keras.utils.Progbar(len(vectors) // batch_size)
	vectors = tf.convert_to_tensor(vectors)
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
		    file_path = os.path.join(dir_out, f'{img_no:06}.png')
		    image = tf.convert_to_tensor(image)
		    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
		    image = tf.io.encode_png(image).numpy()
		    with open(file_path, 'wb') as f:
		        f.write(image)


def get_subdir_out(
	kind,
	dir_out,
	count=None,
	segments=None,
	frames=None):
	instance = f'{time.time()}'
	if kind == 'random':
		subdir = f'{kind}-{count}'
	if kind == 'bezier':
		subdir = f'{kind}-s{segments}-f{frames}'
	if kind == 'sine':
		subdir = f'{kind}-f{frames}'
	subdir_out = f'{dir_out}/{subdir}/{instance}'
	os.makedirs(subdir_out, exist_ok=True)
	return subdir_out


def random(
	model,
	dir_out,
	count):
	generator = load_generator(model)
	subdir_out = get_subdir_out(
		'random',
		dir_out,
		count=count) 
	noise_shape = generator.input_shape[-1]
	vectors = tf.random.normal((count, noise_shape))
	generate(vectors, subdir_out)


def bezier_interpolation(p0, p1, p2, p3, p4, frames):
	segment = []
	t_intervals = tf.linspace(0.0, 1.0, frames)
	for t in t_intervals:
		B_t = (
		    1 * (1 - t)**4 * t**0 * p0 +
		    4 * (1 - t)**3 * t**1 * p1 +
		    6 * (1 - t)**2 * t**2 * p2 +
		    4 * (1 - t)**1 * t**3 * p3 +
		    1 * (1 - t)**0 * t**4 * p4
		)
		segment.append(B_t)
		#print(f'segment shape: ({len(segment)}, {len(segment[0])})')
	return segment


def bezier(
	model,
	dir_out,
	frames,
	segments):
	generator = load_generator(model)
	subdir_out = get_subdir_out(
		'bezier',
		dir_out,
		segments=segments,
		frames=frames) 
	noise_shape = generator.input_shape[-1]
	noises = tf.random.normal((segments * 3, noise_shape))
	vectors = []
	prog_bar = tf.keras.utils.Progbar(segments)
	for i in range(segments):
		p0 = noises[3*i-1]
		p1 = p0 + (p0 - noises[3*i-2])
		p2 = noises[3*i]
		p3 = noises[3*i+1]
		p4 = noises[3*i+2]
		vectors.extend(bezier_interpolation(p0, p1, p2, p3, p4, frames))
		prog_bar.add(1)
	#print(f'vectors shape: ({len(vectors)}, {len(vectors[0])})')
	generate(vectors, subdir_out)


def sine_phases(seedvec, sec=60, fps=30):
	# every dimension has its own wave
	# they have same period length and different phases
	dims = len(seedvec[0])
	waves = []
	for dim in range(dims):
		wave = sinewaves(seedvec[0][dim], sec*fps, 1)
		wave = np.roll(wave, r.randint(1, sec*fps))
		waves.append(wave)
	return waves


def sine(
	model,
	dir_out,
	frames):
	generator = load_generator(model)
	dims = generator.input_shape[-1]
	subdir_out = get_subdir_out(
		'sine',
		dir_out,
		frames=frames)
	os.makedirs(subdir_out, exist_ok=True)
	seedvec = np.random.normal(size=(1, dims))
	waves = sine_phases(seedvec)
	generate(np.transpose(waves), subdir_out)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--model',
		type=str,
		required=True,
		help='model used to generate images')
	parser.add_argument(
		'--dir_out',
		type=str,
		default=f'out/random/{time.time()}',
		help='output folder')
	parser.add_argument(
		'--count',
		type=int,
		default=100,
		help='number of images to generate')
	args = parser.parse_args()
	random(args.dir_in)















