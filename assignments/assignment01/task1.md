# Overview

I've set up a GPU coding environment using Google Colab with NVIDIA CUDA.

# Environment Details

- _GPU Instance_: Tesla T4 (via Google Colab free tier)
- _CUDA_: Version 12.4 (driver), 12.5 (toolkit)
- _Compiler_: `nvcc` (NVIDIA CUDA Compiler)
- _Repo Link_: https://github.com/abdullahejazjanjua/gpu-programming

# My Work Pipeline

1. I write code locally using my IDE
2. Optionally test the `.cu` file using [leetgpu](https://www.leetgpu.com/)
3. I then use [rclone](https://rclone.org/) to sync my work folder with the folder I created in Google Drive as:

```
bash
rclone sync ./<work-directory-to-sync> <remote-name>:/<folder-to-sync-to>/<work-directory-to-sync-to>
```

4. In Google Colab, I mount the drive, navigate to the work folder, and compile the code using:

```
bash
!nvcc -arch=sm_75 <gpu-kernel>.cu <main-file>.cpp -o <executable-name>
```

> You can find more details about how I came to this solution [here](https://www.shashankshekhar.com/blog/cuda-colab)

- The `-arch=sm_75` is passed to ensure that the generated executable is runnable on the T4 GPU that Colab provides in the free tier.
- An example notebook can be viewed [here](https://github.com/abdullahejazjanjua/gpu-programming/blob/mai/interface.ipynb).
