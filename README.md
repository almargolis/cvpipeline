# cvpipeline

OpenCV vision pipeline and filter system extracted from the VNAVS robotics framework.

Provides chainable image filters, color space tracking, geometry primitives, and a Tkinter-based GUI editor for building computer vision pipelines.

## Process Scripts

Setting up a filter pipeline takes time, and you may need to rebuild the same pipeline repeatedly. The GUI has **Open Process** and **Save Process** buttons to save and reload pipelines as `.cvp` script files.

Example scripts are in `scripts/`:

| Script | Description |
|--------|-------------|
| `hough.cvp` | Color mask, erode, Hough line detection |
| `test.cvp` | Minimal pipeline: capture and grayscale |
| `contours.cvp` | Gaussian blur, Canny edge detection, contour finding |

### Script format

Each script is a text file. Filter names are prefixed with `/`, followed by `parm.name=value` lines:

```
/Image
/ColorMaskSingle
parm.ColorMaskSingle_hue=163
parm.ColorMaskSingle_hueRange=25
/MorphErode
parm.MorphErode_kernel_dim=5
parm.MorphErode_iterations=1
```

See `load_process_file()` in `cvpipeline.py` for parsing details.
