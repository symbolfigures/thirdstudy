import argparse
import ctypes
import numpy as np
import os
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit

# load CUDA library
blend_lib = ctypes.CDLL('./blend.so')

# define function signature
blend_lib.blendBatchCUDA.argtypes = [
	ctypes.POINTER(ctypes.c_ubyte),
	ctypes.POINTER(ctypes.c_ubyte),
	ctypes.c_int,
	ctypes.c_int,
	ctypes.c_int
]

def main(dir_in, dir_out):
	filenames = os.listdir(dir_in)
	os.makedirs(dir_out, exist_ok=True)

	# determine batch size
	(w, h) = Image.open(f'{dir_in}/{filenames[0]}').size
	im_mem = 2 * (w * h * 3)
	free_mem = cuda.mem_get_info()[0]
	b_size = int((free_mem / im_mem) * 0.8)
	num_b = (len(filenames) + b_size - 1) // b_size
	#print(f'\nim_mem: {im_mem}\nfree_mem: {free_mem}\nb_size: {b_size}\nnum_b: {num_b}\n')

	for b_idx in range(num_b):
		b_files = filenames[b_idx * b_size:(b_idx + 1) * b_size]

		# load images
		im_arrs = []
		for filename in b_files:
			im = Image.open(os.path.join(dir_in, filename))
			im_arr = np.array(im, dtype=np.uint8)
			im_arrs.append(im_arr)

		h, w, c = im_arrs[0].shape

		# i/o buffers
		b_in = np.concatenate([im.ravel() for im in im_arrs]).astype(np.uint8)
		b_out = np.zeros_like(b_in, dtype=np.uint8)

		# pointers
		in_ptr = b_in.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
		out_ptr = b_out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

		# call CUDA function
		blend_lib.blendBatchCUDA(in_ptr, out_ptr, w, h, len(b_files))

		# save
		b_out = b_out.reshape(len(b_files), h, w, c)
		for i, filename in enumerate(b_files):
			out_im = Image.fromarray(b_out[i], mode='RGB')
			out_im.save(f'{dir_out}/{filename}')


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
		default='out_blend',
		help='output folder')
	args = parser.parse_args()
	main(args.dir_in, args.dir_out)




