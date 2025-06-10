import argparse
import ffmpeg
import os


def png_to_mp4(dir_in):
	dir_out = f'{dir_in}.mp4'
	(
		ffmpeg
			.input(f'{dir_in}/*.png', pattern_type='glob', framerate=32)
			.output(dir_out, vcodec='libx264', pix_fmt='yuv420p')
			.run()
	)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir_in',
		type=str,
		default='out',
		help='input folder')
	args = parser.parse_args()
	png_to_mp4(dir_in)















