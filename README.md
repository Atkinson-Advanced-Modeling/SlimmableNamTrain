# Slimmable NAM Training
Training code for slimmable NAMs

## Installation
This is an [extension](https://github.com/sdatkinson/neural-amp-modeler/pull/441) for [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler).
To use it,
1. Clone this repo and and install the package (i.e. `pip install .`).
2. Place `extension_slimmable.py` in your extensions folder.

This will enable you to use the `SlimmableWaveNet` model architecture when training using `nam-full`.

## Data
To replicate results from the paper, download data from [Google Drive](https://drive.google.com/drive/folders/1XFx1XLU0x-f84B16uMqXuzDiBrLtiqav?usp=drive_link) to `inputs/data/`.

## Training a model
Use the `nam-full` entry point from [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler).
Examples of configuration files for the new architecture are available in the `inputs/models/` directory.

Then, run `nam-full` as usual from the command line, e.g.
```bash
nam-full \
inputs/configs/data/clean.json \
inputs/configs/models/a2/slimmable.json \
inputs/configs/learning.json \
outputs/clean
```
Note that relative paths are used in the data configurations, so you should run from the directory containing this README.

## Roadmap
I intend to integrate this (or something essentially similar) to the main repo for slimmable models to be an easy-to-use modeling approach with NAM.
After that, this extension will be deprecated in favor of that feature.
The specifics of the architecture configuration are still being explored; feedback is welcome!
