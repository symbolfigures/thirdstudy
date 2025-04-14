#include <cuda_runtime.h>
#include <iostream>

extern "C" void blendBatchCUDA(unsigned char *input, unsigned char *output, int w, int h, int batch_size);

__global__ void blendKernelBatch(unsigned char *input, unsigned char *output, int w, int h, int batch_size) {
	int batch_idx = blockIdx.z;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= w || y >= h || batch_idx >= batch_size) return;

	int im_offset = batch_idx * w * h * 3;
	int sumR = 0, sumG = 0, sumB = 0;
	int count = 0;

	// interate through the 3x3 grid
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			int nx = x + dx;
			int ny = y + dy;

			// check bounds
			if (nx >= 0 && ny >= 0 && nx < w && ny < h) {
				int idx = (ny * w + nx) * 3;
				sumR += input[idx];
				sumG += input[idx + 1];
				sumB += input[idx + 2];
				++count;
			}
		}
	}

	// average
	int idx = im_offset + (y * w + x) * 3;
	output[idx] = sumR / count;
	output[idx + 1] = sumG / count;
	output[idx + 2] = sumB / count;
}

extern "C" void blendBatchCUDA(unsigned char *input, unsigned char *output, int w, int h, int batch_size) {
	unsigned char *d_input, *d_output;
	size_t im_size = w * h * 3;
	size_t batch_size_bytes = im_size * batch_size;

	cudaMalloc(&d_input, batch_size_bytes);
	cudaMalloc(&d_output, batch_size_bytes);
	cudaMemcpy(d_input, input, batch_size_bytes, cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y, batch_size);
	blendKernelBatch<<<gridDim, blockDim>>>(d_input, d_output, w, h, batch_size);

	cudaMemcpy(output, d_output, batch_size_bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}









