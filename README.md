## HDR_Viewer_PyPlot
A simple high dynamic range (HDR) image viewer using matplotlib.pyplot for displaying and imageio for reading/writing. It's all in python, easy to extend. 

![alt text](https://github.com/chalmersgit/HDR_Viewer_PyPlot/blob/master/HDR_Viewer_PyPlot_Example.png?raw=true)

## Features
- Displays HDR images (e.g., .exr and .hdr images). I use imageio for reading images, so anything imageio supports can be displayed.
- Toggle gamma correction (g)
- Adjust exposure (scroll). Hold shift to change the exposure faster.
- See pixel values under the mouse cursor
- See general statistics (original resolution, min/max pixel value, spherical mean, spherical standard deviation, contrast)
- Toggle between a diffusely convolved image and the original image (d). Make sure you do this for HDRI's intended for image-based lighting. Note that the first time you press d, it will take a few seconds to compute the diffuse image.
- Save your edited image (o), saves to ./output/ or ./dist/HDR_Viewer_PyPlot/output/

## Running via python
You'll need to pip install modules including imageio [(with EXR support)](https://imageio.readthedocs.io/en/v2.8.0/format_exr-fi.html)
, matplotlib, numpy, and cv2.

The run:
python [string filepath] optional: [int resizeWidth]

For example:

`python ./images/grace-new.exr 512`

resizeWidth is useful if you have very large images that will be too slow to display.

## Packaging into an Executable
It can be useful to compile into an executable, so then you can have your .exr or .hdr images always open with this program.

`pip install pyinstaller`

`pyinstaller HDR_Viewer_PyPlot.spec --noconfirm`

This will package into dist/HDR_Viewer_PyPlot. Point your HDR images (i.e., right click the file, "Open with" > "Choose another app", and select dist/HDR_Viewer_PyPlot/HDR_Viewer_PyPlot.exe

When rebuilding, you might need to manually delete dist in case of permission denied issues (e.g., `rmdir /s dist` and maybe `rmdir /s build`).
