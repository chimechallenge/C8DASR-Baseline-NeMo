## Error: CUDNN_STATUS_NOT_INITIALIZED

`Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED`  
`libcudnn_cnn_train.so.8 Error`

This error is typically due to a version mismatch between libcudnn8's CUDA version and the CUDA toolkit version. To resolve this, you need to match the CUDA version you see from `nvcc -V` with the version of libcudnn8 you install.

### Steps to Resolve

1. Check your CUDA version with the following command:

    ```
    nvcc -V
    ```

2. List all available versions of libcudnn8 and find the one that matches your CUDA version:

    ```
    sudo apt list -a libcudnn8
    ```

3. Install the appropriate version of libcudnn8. For example, if your CUDA version is 12.1 and the latest libcudnn8 version for CUDA 12.1 is 8.9.3.28-1, you would use the following command:

    ```
    sudo apt-get install libcudnn8-dev=8.9.3.28-1+cuda12.1
    ```

This command will purge the old libcudnn and reinstall the new, specified one.

### Manual Installation of CUDNN

If the above steps do not resolve the issue, you can manually download and install the CUDNN version that fits your OS and CUDA version.

1. Go to the [CUDNN website](https://developer.nvidia.com/rdp/cudnn-download) and download the appropriate version for your OS and CUDA version. You can check your OS version with the following command:

    ```
    lsb_release -a
    ```

2. Follow the Debian Local Installation instructions provided in the [NVIDIA cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb). This involves using `dpkg` to install the downloaded files, copying the CUDA GPG key, and installing the runtime library, developer library, and samples.