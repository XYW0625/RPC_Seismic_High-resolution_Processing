# RPC_Seismic_High-resolution_Processing

This repository provides a compact demo implementation of a reflection-position constrained seismic high-resolution processing method. The algorithm combines sparse-spike deconvolution, multi-scale CNN-based reflectivity prediction, and reflection-position constraint information to recover high-resolution reflection coefficients from band-limited seismic data.

The RPC method uses predicted reflection-position masks to guide seismic inversion, suppress non-reflection artifacts, and improve the spatial continuity and resolution of reconstructed reflectivity sections. The repository includes runnable Jupyter Notebook demos for SSD, neural-network prediction, and RPC-based constrained processing, together with synthetic `.mat` seismic datasets and plotting utilities for paper-style wiggle-trace visualization.

## Main Features

- Sparse-spike deconvolution baseline for seismic high-resolution reconstruction.
- Multi-scale CNN model for reflectivity prediction from seismic traces.
- Reflection-position constrained processing to enhance inversion reliability.
- Synthetic seismic data examples in MATLAB `.mat` format.
- Reproducible Jupyter Notebook demos and visualization scripts.
- Compact public demo setup for quick testing and comparison.
