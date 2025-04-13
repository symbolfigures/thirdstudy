import argparse
import ffmpeg
from generate import bezier, sine


def png_to_mp4(dir_in):
	dir_out = f'{dir_in}.mp4'
	(
		ffmpeg
			.input(f'{dir_in}/*.png', pattern_type='glob', framerate=32)
			.output(dir_out, vcodec='libx264', pix_fmt='yuv420p')
			.run()
	)


def main(args):
	style, model, dir_out, frames, segments = args

	if style = 'bezier':
		bezier(model, dir_out, frames, segments)
	if style = 'sine':
		sine(model, dir_out, frames)

	png_to_mp4(dir_out)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--style',
		type=str,
		required=True,
		choice=['bezier', 'sine'],
		help='either bezier or sine')
	parser.add_argument(
		'--model',
		type=str,
		required=True,
		help='model used to generate images')
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

	args = parser.parse_args()
	main(args)














