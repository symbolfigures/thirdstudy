## Create the dataset

Enter `data`.  
`$ cd data`

The network needs a lot of data to train on, and drawing can take a long time, so it's convenient if the scanned images can be cropped, flipped, or rotated. That way the network gets more data for the same price. Drawings that are very abstract are suitable in this regard; a region may be cropped, flipped, and rotated arbitrarily without losing value. For comparison, a portrait or still life can only be flipped horizontally, and cropped only to remove the background. The former kind I call a blob, and the latter a specimen.

If you don't have your own media and want to follow along, you can download my own drawing scans, and then move them to the `data/scan/originals` folder and unzip. ([second study](https://symbolfigures.io/thirdstudy/data/scan/originals/secondstudy_dpi300.zip)) ([third study](https://symbolfigures.io/thirdstudy/data/scan/originals/thirdstudy_dpi300.zip))

Third Study is a kind of blob, and Second Study is a kind of specimen. Pick one, copy it to the `data` folder, and give it a unique name. In this walkthrough, we will use `web` and `dilly` for third and second studies respectively, and default to `web` for operations required by both blobs and specimens.

Enter `scan`, copy the originals, and rename them. 
`$ cd scan`  
`$ cp -r originals/thirdstudy_dpi300 .`  
`$ mv thirdstudy_dpi300 web`

### 1. Prepare the drawing scans

If the images aren't in PNG format, convert them to PNG. Specify the folder at the end of the command.  
`$ bash jpeg_to_png.sh web`

If the images aren't grayscale, convert to grayscale. The drawings are colorless enough, and this will save resources during training.  
`$ python rgb_to_gray.py web`

They need to be named `01.png`, `02.png`, ... for further processing.  
`$ bash rename.sh web`

Exit `scan`  
`$ cd ../`

### 2. Cut lots of tiles

`tile.py` takes the scans and produces thousands of cropped images for the neural network to train on.

#### 2.1 Blob

Every drawing in a blob is like a block of clay that can be cut this way and that, into cubes of any size. Still, care must be taken to stay inside of the margin. Drawings have varying margins, so the `grid` function helps define the right margin for each drawing.

It creates a `grid` folder with images inside. These are the same scans with a grid superimposed. Each cell of the grid covers 256/300 square inches, and is scaled depending on dpi. The default grid size is 12 x 18 cells.

Run `grid` with `-h` to see all the available parameters.  
`$ python tile.py grid -h`

Run `grid`, and pass in the dpi and input directory. Review each page and see if any of the grids need to be shifted up, down, left or right.  
`$ python tile.py grid 300 web`

The `adj_xy.json` file contains adjustments for each page, which you may edit manually. It's already adjusted for the third study. The values are measured in `unit`s and indicated by page number. `adj_x` is horizontal, and `adj_y` is vertical. If you just need to adjust a single page, run `grid` again with the page number.  
`$ python tile.py grid 300 web --page=01`

The `blob` function in `tile.py` produces the sample images to train on. It takes two additional parameters:

- `pixels`: Square pixels each tile will have. It must be a power of 2 between 4 and 1024. This is the resolution of the animation.
- `steps`: Inverse of the fraction of a unit that adjacent tiles are separated by. The higher the number, the more they overlap. If dpi/pixels are of the proportion 300/256; then 1 step means adjacent tiles don't overlap, 2 steps overlap by 1/2 unit, 3 steps overlap by 2/3 unit, and so on.

For example, with 46 drawings at 11 x 16 inches, 300 dpi scans, 512x512 pixels per tile, and a step count of 16, `blob` cuts 1,643,166 tiles (286 GiB).  
`$ python tile.py blob 300 web --pixels=512 --steps=16`

Remark: It's tempting to increase step count to generate so many images that the neural network never sees the same image twice. However, a high step count results in animations where the picture tends to spin around without the lines moving.

#### 2.2 Specimen

The `specimen` function enables the user to cut tiles individually at a specified resolution. It displays one drawing at a time, and the user can click at the center of the desired tile. For every click, an image is saved in the `tile` folder. This works for a set of drawings that have a varying number of specimens per page.

Pass in the dpi, pixels, and page number. Use arrow keys to scroll.  
`$ python tile.py specimen 300 dilly --pixels=512 --page=01`

After that, `rotateflip` turns each tile into 8 tiles by rotating at 90 degree intervals, and flipping each rotation. No need to pass in the dpi.  
`$ python tile.py rotateflip dilly`

A set of 43 drawings with 36 specimens each yields a total of 12,384 tiles (2 GiB) to train on.

### 3. Create .tfrecord files

The last step before training is to convert the tiles into shards, or .tfrecord files, which is the format used by the neural network. The maximum shard size has a default value of 500 MB.  
`$ python tfrecord.py web`

Exit `data`  
`$ cd ../`