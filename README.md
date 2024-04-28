# SfM and NeRF Pipelines for 3D Scene Understanding

This repository contains a computer vision project implementing Structure from Motion and NeRF

## Running the Software

To run the software, make sure the following modules are installed (also contained in `requirements.txt`):
- argparse
- matplotlib
- numpy
- opencv-python
- scipy

The main files are:
- `utils.py`
- `get_dataset_info.py`
- `main.py`
- `pipeline.py`

Run the software with `main.py -dataset=<dataset>`. Use the `-dataset` flag to specify the dataset (an integer).

## Reconstruction Results

Reconstructions for each dataset before and after LM optimization are available in the repository. The reconstructions visually improve after optimization, aligning point clouds more accurately.

## Running the Tiny NeRF

To run the Tiny NeRF, you need to create a project on wandb and get the API key. Then, run the following command:
Note using wandb is optional and you can run the code without it just by commenting out the wandb related code.

Now just execute the Jupyter notebook `tiny_nerf.ipynb` to run the Tiny NeRF. The notebook will train the Tiny NeRF on the dataset and display the results.
