import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import numpy as np
import os
from PIL import Image, ImageDraw
from random import randrange, shuffle
import sys
import time

sys.setrecursionlimit(10000000)


def get_colors():
	img = Image.open('palet.png')
	colors = []
	for x in range(img.width):	
		for y in range(img.height):
			colors.append(img.getpixel((x, y)))
	return colors


def color_match(pix, overlay):
	palet = Image.open(overlay)
	colors = []
	for p in pix:
		colors.append(palet.getpixel(p))
	R = 0
	G = 0
	B = 0
	for c in colors:
		R += c[0]
		G += c[1]
		B += c[2]
	R = R // len(colors)
	G = G // len(colors)
	B = B // len(colors)
	return (R, G, B)


def bitmap(img):
	img = img.convert('L')
	array = np.array(img)
	array = np.where(array > 128, 255, 0).astype(np.uint8)
	return Image.fromarray(array.astype(np.uint8))


def get_shape(point, img, draw):
	(w, h) = img.size
	(x, y) = point
	pix = []
	pix_set = set()

	def gather_pix(x, y):
		if img.getpixel((x, y)) != (255, 255, 255):
			return
		pix.append((x, y))
		pix_set.add((x, y))
		if x > 0 and (x - 1, y) not in pix_set:
			gather_pix(x - 1, y)
		if x < w - 1 and (x + 1, y) not in pix_set:
			gather_pix(x + 1, y)
		if y > 0 and (x, y - 1) not in pix_set:
			gather_pix(x, y - 1)
		if y < h - 1 and (x, y + 1) not in pix_set:
			gather_pix(x, y + 1)

	gather_pix(x, y)
	return pix


def neighborhood(q, img):
	# is every pixel in the neighborhood white
	(w, h) = img.size
	(x, y) = q
	neighbors = []
	neighbors.extend([
		(x-2, y),
		(x-1, y-1), (x-1, y), (x-1, y+1),
		(x, y-2), (x, y-1), (x, y), (x, y+1), (x, y+2),
		(x+1, y-1), (x+1, y), (x+1, y+1),
		(x+2, y)
	])
	for (xi, yi) in neighbors:
		if 0 <= xi < w and 0 <= yi < h:
			if img.getpixel(p) != (255, 255, 255):
				return False
				break
	return True


def fill_line(x, y, img, draw):
	(w, h) = img.size
	z = 0
	while True:
		z += 1
		box = [
			(x - z, y - z),
			(x, y - z),
			(x + z, y - z),
			(x - z, y),
			(x + z, y),
			(x - z, y + z),
			(x, y + z),
			(x + z, y + z)]
		shuffle(box)
		for (xi, yi) in box:
			if 0 <= xi < w and 0 <= yi < h:
				color = img.getpixel((xi, yi))
				if color != (0, 0, 0):
					draw.point((x, y), fill=color)
					return


def blend_px(x, y, img):
	(w, h) = img.size
	box = [
		(x-1, y-1), (x-1, y), (x-1, y+1),
		(x, y-1), (x, y), (x, y+1),
		(x+1, y-1), (x+1, y), (x+1, y+1)]
	r, g, b = 0, 0, 0
	for (xi, yi) in box:
		if 0 <= xi < w and 0 <= yi < h:
			c = img.getpixel((xi, yi))
		else:
			c = img.getpixel((x,y))
		r += c[0]
		g += c[1]
		b += c[2]
	color = (r//9, g//9, b//9)
	return color


def process(args):
	dir_in, dir_out, blend, overlay, filename = args

	img = Image.open(f'{dir_in}/{filename}')
	img = bitmap(img).convert('RGB')
	draw = ImageDraw.Draw(img)
	(w, h) = img.size

	# fill space
	for x in range(w):
		for y in range(h):
			if img.getpixel((x, y)) == (255, 255, 255):
				pix = get_shape((x, y), img, draw)
				if overlay is not None:
					color = color_match(pix, overlay)
				else:
					color = colors[randrange(len(colors))]
				if pix != []:
					for p in pix:
						draw.point(p, fill=color)


	# fill line
	for x in range(w):
		for y in range(h):
			if img.getpixel((x, y)) == (0, 0, 0):
				fill_line(x, y, img, draw)

	# blend
	if blend:
		trace = img
		for x in range(w):
			for y in range(h):
				color = blend_px(x, y, trace)
				draw.point((x, y), color)

	img.save(f'{dir_out}/{filename}')


def main(dir_in, dir_out, blend, overlay):
	if not overlay:
		colors = get_colors()
	os.makedirs(dir_out, exist_ok=True)
	filenames = os.listdir(dir_in)
	args = [(dir_in, dir_out, blend, overlay, x) for x in filenames]
	with ProcessPoolExecutor() as executor:
		executor.map(process, args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir_in',
		type=str,
		required=True,
		help='path to source images folder')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='out',
		help='output folder')
	parser.add_argument(
		'--blend',
		action='store_true',
		default=False,
		help='blend with no GPU support')
	parser.add_argument(
		'--overlay',
		type=str,
		default=None,
		help='pick colors according to underlying image. provide the path.')
	args = parser.parse_args()
	main(args.dir_in, args.dir_out, args.blend, args.overlay)

























