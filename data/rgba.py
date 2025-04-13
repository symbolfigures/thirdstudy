# source material: scans in RGB
# convert each image file to RGBA
# induce transparency as a measure of whiteness
# save using naming convention 0.png, 1.png, ...
#
# on color modes:
#
# converting to grayscale 'L' seems efficient by requiring only 1 channel
# but transparency is desired, and only obtained with RGBA which keeps the 3 from RGB
# without transparency, grayscale is a good option
# there are 2 places to update the channel count in models.py
#
# converting to bitmap '1' or forcing values as {0, 1}
# seems efficient but the data structure in the GAN is still float32 or mixed float16
# takes no advantage of the memory reduction
#
# forcing values as {0, 255} by first converting to '1' and then to 'RGBA'
# which is convenient for downstream processing
# the resulting images and latent space are too homogeneous:
# images look like tessellations, and animations move very slowly
# the loss of information appears significant to the GAN if not to the eye
# gives it fewer clues to find the minima
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
from PIL import Image


def worker(args):
	i, filepath, dir_out = args
	im = Image.open(filepath)
	# RGB: 3 channels, range 0-255
	# L: 1 channel, range 0-255
	# 1: 1 channel, range {0, 1}
	# RGBA: 4 channels, range 0-255
	assert im.mode == 'RGB', 'source images must be in RGB'

	# set alpha so lighter color is more transparent
	im = im.convert('RGBA') # every pixel gets a 4th channel with value 255
	opaque = im.getdata()
	transparent = []
	for px in opaque:
		avg = (px[0] + px[1] + px[2]) / 3
		alpha = 255 - int(avg)
		transparent.append((px[0], px[1], px[2], alpha))
	im.putdata(transparent)

	im.save(f'{dir_out}/{i}.png')


def main(dir_in):
	dir_out = f'{dir_in}_rgba'
	os.makedirs(dir_out, exist_ok=True)
	filepaths = [f'{dir_in}/{f}' for f in os.listdir(dir_in)]
	args = [(i, f, dir_out) for i, f in enumerate(filepaths)]
	with ProcessPoolExecutor() as executor:
		executor.map(worker, args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir_in',
		type=str,
		required=True,
		help='folder of scans e.g. scan/field_dpi300')
	args = parser.parse_args()
	main(args.dir_in)





















