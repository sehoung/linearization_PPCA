# Linearized Regional Seismic Loss Estimation with PPCA

This software provides Python code to simulate **regional seismic loss estimation**. The software is also available through DesignSafe: https://doi.org/10.17603/ds2-9hbb-c610

It uses the computational framework proposed by **Houng and Ceferino (2026)**, which:

1. Linearizes the **ground-motion–fragility coupling**
2. Incorporates **Probabilistic Principal Component Analysis (PPCA)** for dimensionality reduction, enabling more efficient simulations.

This approach reduces the computational complexity from **O(N³)** in traditional frameworks to **O(N²)**.

The framework has been validated using a building portfolio in **San Francisco**, demonstrating high accuracy within the stochastic error range of **Monte Carlo (MC)** simulations:

- **>95% agreement** in modal damage state  
- **<2.5% error** in loss estimation  

For large portfolios containing approximately **30,000 buildings**, the method achieves about **110× faster simulations** compared to conventional testbeds.

---

# How to Run the Code

Below is a simple instruction to run the code.

## Directory Structure

## Input Data

`data/` contains the input data for the simulation:

- **SanFrancisco_buildings_ext.csv**  
  Target building inventory composed of ~15K buildings in downtown San Francisco.

- **fragility_PGA.csv**  
  Fragility curves depending on structural types.

- **consequence_repair_PGA.csv**  
  Repair cost and repair time associated with each damage state.

## Code Files

`code/` contains scripts for running simulations and generating figures:

- **run_proposed.py**  
  Runs the proposed framework.

- **run_traditional.py**  
  Runs the traditional framework.

- **run_figure_preprocessing.py**  
  Preprocesses the output from `run_proposed.py` and `run_traditional.py`.

- **run_figure.py**  
  Generates figures comparing the loss curves from the traditional and proposed frameworks.

## Output

Simulation results and figures will be saved in: `out/figs`
