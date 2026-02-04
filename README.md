# GPU Programming with CUDA

## Overview

This repo contains CUDA code for GPU programming course at GIKI.
I use Google Colab for running CUDA code with rclone for syncing local code to Colab.

## Setup

- Write code locally
- Use rclone to sync to Google Drive/Colab
- Run & compile in `interface.ipynb` on Colab

## How to Run

See `interface.ipynb` for steps to:

1. Test kernel using leetgpu
2. Sync code using rclone

```bash
rclone sync ./<directory-to-sync> <remote-name>:/<folder-to-sync-to>/<directory-to-sync-to>
```

3. Compile CUDA code
4. Run on Colab GPU
