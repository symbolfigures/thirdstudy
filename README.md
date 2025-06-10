# Third Study

Scripts that 
- prepare scanned ink drawings as a dataset for deep learning.
- generate immitation images from a trained model.
- fill generated images with color to create a synthetic dataset.
- make animations by calculating transformations through the model's latent space.

The drawings I use are from my [Third Study](https://symbolfigures.io/thirdstudy.html). Generally, comptible drawings
- fill the page within some rectangular margin.
- may be flipped, and rotated by any degree.
- have dark lines on a light background.

The model is trained with a generative adversarial network (GAN). The code for the GAN is originally from a [course](https://www.udemy.com/course/high-resolution-generative-adversarial-networks) by Brad Klingensmith, which is in turn based off [ProGAN](https://arxiv.org/abs/1710.10196) with improvements from [StyleGAN2](https://arxiv.org/abs/1912.04958).

Scripts in this repo which are only derivations of the GAN are located at:
- [train/src/](train/src/)
- [anim/src/gan/](anim/src/gan/)
- [anim/src/train.py](anim/src/train.py)

The remaining scripts were made specifically for the purpose of animating the Third Study drawings.

What follows are the basic steps to run the scripts. Further comments are in the code.

## Setup

Clone and enter the repo using your preferred method. Create a virtual environment and install the required packages.

```
git clone git@github.com:symbolfigures/thirdstudy.git
cd thirdstudy
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, freeze.txt has all the dependencies and versions explicit.

## Data

`cd data`

The scans should be prepared as PNG files in RGB mode.

Adjust grid placement.  
`python grid.py --dir_in="<folder_path>" --dpi=300`

Create images for training data.  
```
python tile.py \
	--dir_in="scan/field_dpi300_rgba" \
	--dpi=300 \
	--resolution=1024
```

ex. [tile](https://symbolfigures.io/thirdstudy/demo/tile.png)

Convert .png to .tfrecord.  
`python tfrecord.py --dir_in="<folder_path>"`

Verify .tfrecord data by extracting .png.  
`python tfrecord_reverse.py --dir_in="<folder_path>"`

`cd ../`

## Train

Includes a reduced version of the GAN from the course.

`cd train`

Train the model, passing parameters in ./params file.
```
export TF_USE_LEGACY_KERAS=1
python src/main_training.py "@params"
```

`cd ../`

## Generate

Includes a further reduced GAN only capable of inference.

`cd gen`

Generate random images for sythetic datasets.  
`python src/generate.py --model="<model_path>"`

ex. [random](https://symbolfigures.io/thirdstudy/demo/random.png)

Bezier: continuous transformation follows a looping bezier curve passing through the model's _n_-dimensional latent space.  
`python src/anim.py --style="bezier" --model="<model_path>"`

ex. [bezier](https://symbolfigures.io/thirdstudy/demo/bezier.mp4)

Sine: dimensions of the latent space are modulated by a sine wave. The sine waves have constant period and variable phases.  
`python src/anim.py --style="sine" --model="<model_path>"`

`cd ../`

## Fill

`cd fill`

Given a set of random generated images,
- fill the shapes with color
- remove the lines in between

`python fill.py --dir_in="<folder_path>"`

ex. [fill](https://symbolfigures.io/thirdstudy/demo/fill.png)

The `--blend` option blends the edges, but is kind of slow. To accomplish the same with a GPU, use [blend.py](fill/blend.py) instead. First, compile [blend.cu](fill/blend.cu).

```
nvcc -shared -o blend.so blend.cu --compiler-options '-fPIC'
python blend.py --dir_in="<folder_path>"
```

ex. [blend](https://symbolfigures.io/thirdstudy/demo/blend.png)

The `--overlay` option includes a reference image to pick colors from, according to the shape's location relative to the image.

`python fill.py --dir_in="<folder_path>" --overlay="<image_path>"`

ex. [reference](https://symbolfigures.io/thirdstudy/demo/reference.png), [overlay](https://symbolfigures.io/thirdstudy/demo/overlay_random.png), [over a bezier curve](https://symbolfigures.io/thirdstudy/demo/overlay_bezier.mp4)

`cd ../`























