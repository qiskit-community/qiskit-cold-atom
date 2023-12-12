# Benchmarks

This directory contains benchmarks designed to be run using [asv](https://asv.readthedocs.io/). To run these benchmarks:
1. Install dependencies.
```bash
pip install asv virtualenv packaging
```
2. Run the benchmarks from the top level directory of this repository (not the directory containing this file, but one level above it).
```bash
asv run
```