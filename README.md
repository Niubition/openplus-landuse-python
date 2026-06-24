# OpenPLUS Python

Clean-room Python approximation of the PLUS model workflow.

This repository contains a Python implementation of the main PLUS-style
workflow used in this project:

- extract land expansion from two LULC rasters
- train LEAS-like development-potential rasters
- run a CARS-like cellular automata simulation
- validate simulation results with OA, Kappa, confusion matrix, and FoM

## Source and Background

PLUS stands for **Patch-generating Land Use Simulation**. The model combines:

- **LEAS**: Land Expansion Analysis Strategy, used to mine land expansion rules
  from historical land-use change and driving factors.
- **CARS**: CA based on multi-type random patch seeds, used to simulate future
  land-use patterns under demand, neighborhood, transition, and policy
  constraints.

The original PLUS model and software are described in:

> Xun Liang, Qingfeng Guan, Keith C. Clarke, Shishi Liu, Bingyu Wang, Yao Yao.
> 2021. Understanding the drivers of land expansion for sustainable land use
> using a patch-generating land use simulation (PLUS) model: A case study in
> Wuhan, China. Computers, Environment and Urban Systems, 85, 101569.
> https://doi.org/10.1016/j.compenvurbsys.2020.101569

Official project page:

https://github.com/HPSCIL/Patch-generating_Land_Use_Simulation_Model

## What We Did

This code was written as a **clean-room reimplementation** based on public
materials: the PLUS paper, public manuals, parameter files, and sample data
behavior.

It does **not** contain extracted, decompiled, or reverse-engineered code from
the PLUS executable.

The implementation in `openplus.py` provides:

- a PLUS-style land expansion extractor
- a lightweight NumPy random forest for LEAS-like probability surfaces
- a CARS-like cellular automata simulation engine
- validation utilities for overall accuracy, Kappa, confusion matrix, and FoM
- optional calibration by selecting the best simulation iteration using a known
  truth raster

## Requirements

Python packages:

- `numpy`
- `rasterio`

Example check:

```powershell
python -c "import numpy, rasterio; print('ok')"
```

## Usage

Show available commands:

```powershell
python openplus.py --help
```

Extract land expansion:

```powershell
python openplus.py extract-expansion ^
  --start path\to\lulc_start.tif ^
  --end path\to\lulc_end.tif ^
  --out outputs\landexpansion.tif
```

Train LEAS-like probability rasters:

```powershell
python openplus.py train-leas ^
  --params path\to\LEASparameters.tmp ^
  --expansion outputs\landexpansion.tif ^
  --out-prefix outputs\probability.tif
```

Run CARS-like simulation:

```powershell
python openplus.py simulate-cars ^
  --params path\to\CARSparameters.tmp ^
  --probabilities outputs\probability_band_1.tif outputs\probability_band_2.tif ^
  --out outputs\simulation.tif
```

Validate a simulation:

```powershell
python openplus.py validate ^
  --truth path\to\truth.tif ^
  --simulation outputs\simulation.tif ^
  --initial path\to\initial.tif ^
  --out-csv outputs\validation.csv
```

## Example Result From Local Test Data

Using the Wuhan sample workflow in the local PLUS folder, calibration improved
the validation metrics from:

```text
overall_accuracy = 0.748646
kappa            = 0.612996
```

to:

```text
overall_accuracy = 0.775380
kappa            = 0.668121
```

The key implementation fix was treating the bundled PLUS transition matrix as:

```text
rows    = current land-use classes
columns = future land-use classes
```

This orientation made the sample demand vector reachable and improved the CARS
simulation result.

## Limitations

This project is a functional approximation, not a byte-identical clone of the
official PLUS software. Public materials do not specify all implementation
details, such as exact random forest construction, thread scheduling, random
streams, tie-breaking rules, and every nodata edge case.

Use this repository for research, learning, and reproducible experimentation.
For official PLUS behavior, cite and use the official PLUS software and papers.

