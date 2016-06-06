# pyANPD
pyANPD is a high recall automatic number plate detector based on [this Stack Overflow answer](http://stackoverflow.com/a/37523538/2605733). For details of the algorithm, check the answer on Stack Overflow. The detector is robust to orientation. The precision decreases and recall increases if the `edge_density` threshold is decreased.

## Dependencies
1. [OpenCV](http://opencv.org/downloads.html)
2. [Numpy](http://www.numpy.org/)

## Usage

`python pyANPR.py <image_file_path>`

This outputs an image with the same name with `-detected` appended.

### How to tweak it for your dataset?

Change `aspect_ratio_range` (Range of Aspect Ratio for Accepted Rectangles), `area_range` (Range of Area for Accepted Rectangles), and `se_shape` (Shape of Structuring Element for Morphological Closing) to the taste of your own dataset.

## Results
