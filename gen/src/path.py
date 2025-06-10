import argparse
import ffmpeg
from generate import generate, load_generator
import os
import time
import tensorflow as tf


def get_subdir_out(
	kind,
	dir_out,
	count=None,
	segments=None,
	frames=None):
	instance = f'{int(time.time())}'
	if kind == 'random':
		subdir = f'{kind}_{count}'
	if kind == 'bezier':
		subdir = f'{kind}_s{segments}_f{frames}'
	if kind == 'sine':
		subdir = f'{kind}_f{frames}'
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
	generate(generator, vectors, subdir_out)


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
	generate(generator, vectors, subdir_out)
	return subdir_out


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
	generate(generator, np.transpose(waves), subdir_out)
	return subdir_out


def main(style, model, dir_out, frames, segments, count):

	if style == 'random':
		random(model, dir_out, count)
	if args.style == 'bezier':
		bezier(model, dir_out, frames, segments)
	if style == 'sine':
		sine(model, dir_out, frames)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--style',
		type=str,
		required=True,
		choices=['random', 'bezier', 'sine'],
		help='random, bezier or sine')
	parser.add_argument(
		'--model',
		type=str,
		required=True,
		help='path to model used to generate images')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='out',
		help='output folder')
	parser.add_argument(
		'--frames',
		type=int,
		default=256,
		help='number of frames per segment or period')
	parser.add_argument(
		'--segments',
		type=int,
		default=32,
		help='number of points in the vector space for bezier curve to pass through')
	parser.add_argument(
		'--count',
		type=int,
		default=1,
		help='number of random images to generate')

	args = parser.parse_args()
	main(args.style, args.model, args.dir_out, args.frames, args.segments, args.count)














