# VaMiAnalyzer

## Overview
This project consists of two Python programs that together perform image correction and tubular structures analysis on `.tif` format images. The `correct.py` program corrects the images by performing flat-field correction and optionally enhancing local contrast. The `analyze.py` program analyzes the corrected images to identify and process tubular structures.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- BaSiC (Basicpy)
- Matplotlib
- Scikit-image
- NetworkX
- TQDM

## Installation
To run this program, you need to install the required libraries. You can install them using `pip`:

```
pip install opencv-python numpy basicpy matplotlib scikit-image networkx tqdm
```

## Step 1 - Image Shadow Correction
This program uses [BaSic](https://github.com/marrlab/BaSiC) for Background and Shading Correction. This step is for generating good quality images for the image analysis program to process in the next step.

### Parameters
- `input_folder`: Path to the folder containing the input images.
- `output_folder`: Path to the folder where the corrected images and summary file will be saved.
- `contrast_balance` (optional): Boolean indicating whether to apply local contrast enhancement (default is `False`).
- `clip_limit` (optional): Threshold for contrast limiting if local contrast enhancement is applied (default is `2`).
- `tile_grid_size` (optional): Size of the grid for the histogram equalization if local contrast enhancement is applied (default is `8`).

### Usage
1. Ensure that your input folder contains the `.tif` images you want to process.
2. Specify the paths for the `input_folder` and `output_folder` in the script.
3. (optional) This program also provides a contrast_balance method for contrast enhencement. This function is helpful if the images has low local contrast. To apply this method, set `contrast_balance` parameter to `True` and/or set `clip_limit` and `tile_grid_size` value.
4. Run the script below. Square brackets ('[]') are used to denote optional parameters.

```
python correct.py <path/to/input_folder> <path/to/output_folder> [--contrast_balance] [--clip_limit <clip_limit>] [--tile_grid_size <tile_grid_size>]
```

### Output
The corrected images will be saved in the specified output folder with the prefix `corrected_` followed by the original filename. A summary file named `summary.txt` will also be created in the output folder containing the mean and standard deviation of each corrected image.

## Step 2 - Tubular Structures Analysis
This part takes the corrected image from step 1 as its input and analyzes the tubular structures in the images.

### Parameters
  - `input_path`: Path to the input folder or file containing the images. 
  - `output_dir`: Path to the folder where the results will be saved.
  - `mean`: Target mean for image adjustment. After step 1, you may check the corrected images and the summary file. The mean value from an image with good quality could be used as the initial value for this parameter.
  - `std`: Target standard deviation for image adjustment. After step 1, you may check the corrected images and the summary file. The std value from an image with good quality could be used as the initial value for this parameter.
  - `window_size`: Size of the window for binary image creation. A larger window will recognize the texture based on a larger scale.
  - `step_size`: Step size for sliding window in binary image creation. Smaller than window size.
  - `merge_threshold`: Threshold for merging close nodes in the graph. The greater this value, the more distant the nodes that will be merged.
  - `min_tubular_length`: Minimum length of tubular structures to be considered for counting the number of tubes.
  - `min_end_tubular_length`: Minimum length of end tubular structures for pruning.
  - `area_threshold`: Minimum area for regions to be considered in binary image. Regions with an area less than this value will not be considered a valid structure.
  - `std_threshold`: Standard deviation threshold for binary image creation.
  - `radius`: Radius for local region processing in small branch removal. If a node is in the middle of a structure with a radius larger than or equal to this value, the leaf nodes connect to it and their corresponding tubes will be removed.
  - `margin`: Margin to be excluded from binary image creation.

### Usage
1. Ensure that your input path contains the `.tif` corrected images you want to process or specify a single `.tif` image file.
2. Specify the paths for the `input_path` and `output_dir` in the script.
3. Run the script below. Square brackets ('[]') are used to denote optional parameters.

```
python analyze.py <path/to/input> <path/to/output> [--mean <mean_value>] [--std <std_value>] [--window_size <window_size>] [--step_size <step_size>] [--merge_threshold <merge_threshold>] [--min_tubular_length <min_tubular_length>] [--min_end_tubular_length <min_end_tubular_length>] [--area_threshold <area_threshold>] [--std_threshold <std_threshold>] [--radius <radius>] [--margin <margin>]
```

### Output
The processed images with annotated results and statistics will be saved in the specified output folder. The number of loops, branch points, and tubes will be printed out.