# SARI

The Official PyTorch code for ["Semantic-Aware Region Loss for Land-Cover Classification"](https://doi.org/10.1109/JSTARS.2023.3265365)

## Environment

Ubuntu=16.04, CUDA=10.0, python=3.6, Pytorch=1.2.0

2*NVIDIA GeForce RTX 2080Ti

## Installation

1. Install cv2

    `pip install opencv-contrib-python==4.4.0.46`

2. Install pytorch v1.2.0
3. Install InPlace-ABN
    ```
    git clone https://github.com/mapillary/inplace_abn.git
    cd inplace_abn
    python setup.py install
    cd scripts
    pip install -r requirements.txt
    ```
4. Install python packages
    ```
    conda install tqdm
    conda install scipy
    conda install tensorboardX
    ```

## GID Dataset

1. Preprocess (split images and generate semantic superpixels)

    ```
    cd preprocess
    python preprocess.py -d GID -i "xx/GID/image_NirRGB" -l "xx/GID/label_gray" -s "xx/GID/data_1024"
    python datalist.py -d GID -i "xx/GID/data_1024/img" -l "xx/GID/data_1024/sp" -s "xx/GID/data_1024/trainval.lst"
    ```

2. Train model

    ```
    cd ../SPSNet
    bash start_gid_cv.sh
    ```

3. Evaluate model

    `bash eval_gid_cv.sh`

## DeepGlobe Dataset

1. Preprocess (split images and generate semantic superpixels)

    ```
    python preprocess.py -d DeepGlobe -i "xx/DeepGlobe/land-train" -l "xx/DeepGlobe/land-train"  -s "xx/DeepGlobe/data_768"
    python datalist.py -d DeepGlobe -i "xx/DeepGlobe/data_768/img" -l "xx/DeepGlobe/data_768/sp" -s "xx/DeepGlobe/data_768/train.lst" -r "./train_deepglobe.txt"
    python datalist.py -d DeepGlobe -i "xx/DeepGlobe/data_768/img" -l "xx/DeepGlobe/data_768/sp" -s "xx/DeepGlobe/data_768/val.lst" -r "./val_deepglobe.txt"
    python deepglobe_testset.py -i "xx/DeepGlobe/land-train" -l "xx/DeepGlobe/land-train" -p "./test_deepglobe.lst" -s "xx/DeepGlobe/data_test"
    ```

2. Train model

    ```
    cd ../SPSNet
    bash start_deepglobe.sh
    ```

3. Evaluate model

    `bash eval_deepglobe.sh`
