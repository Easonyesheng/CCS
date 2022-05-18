
# Learning-based Camera Calibration
---
This is the official implementation of the paper: 'Learning-Based Framework for Camera Calibration with Distortion Correction and High Precision Feature Detection', by Yesheng Zhang and Xu Zhao and Dahong Qian.

## Abstract
---
We propose a learning-based Camera Calibration Framework (CCF).
In this framework, the accuracy and robustness of camera calibration are improved from three aspects: distortion correction, corner detection and parameter estimation.
Specifically, the distortion correction is performed by the learning-based method.
Accurate feature locations are achieved by the combination of learning-based detection, specially designed refinement and complete post-processing.
Moreover, we obtain stable parameter estimation by a RANSAC-based procedure.

![MainFig](./assets/MainFig.png)

## Paper  
---

The pdf file can be found [here](https://arxiv.org/abs/2202.00158). 

### BibTex
```Latex
@misc{zhang2022learningbased,
    title={Learning-Based Framework for Camera Calibration with Distortion Correction and High Precision Feature Detection},
    author={Yesheng Zhang and Xu Zhao and Dahong Qian},
    year={2022},
    eprint={2202.00158},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## Update log
---
- 22-05-18: Code Release.
- 22-02-01: Code is coming soon...
- 21-06-15: Prepare Demo Data.

## TODO LIST
---

- [x] code release.
- [x] README complete.

## Usage
---

### Requirements
- CUDA ~= 9.2
- python ~= 3.6
- pytorch ~= 1.2.0
- torchvision ~= 0.4.0
- python-opencv ~= 3.4.2
- Numpy ~= 1.16

They can all be installed following command:
``` shell
    conda create -n CCF python=3.6
    pip install -r requirements.txt
    conda activate CCF
```


### File Folder Configuration  


we use the fixed data folder structure for calibration input and output as follows:

``` shell
data
├── DetectRes # [output] for detection
│   ├── color_img # [output] colored heatmap
│   └── heatmap # [output] heatmap
├── GT # [input and optional] for evaluation
├── SubpixelRes # [output] sub-pixel refinement
├── dist_img # [input] original images
└── img # [output or input] corrected images (output) or images without distortion (input)
```

See Examples in `./demo_data/*`.


### Demo data

We provide three sets of calibration data for demo in `./demo_data/`.

For simplicity, we directly provide our distortion correction and detection results here.

You can train your own networks for these results using our training scripts as well.

The images provided here are screened out from a larger image set by our RANSAC-based calibration procedure, for the sake of convenience。

### Run 
First, you need to modify the data path in `./Demo_calib.py `, and you can choose one of the three data sets we provided in `./demo_data/`.

Then the demo calibration can be run following commands:
```python
    python Demo_calib.py
```

The results can be seen like:

![example](./assets/example.png)


### Training

**Note:** Before the training, you need to prepare the dataset following the **Data Generation** part and modify the corresponding parameters in `settings/settings.py`.

We also provide training scripts for our corner detection network and distortion correction network.

*Corner Detection:* You can use the `train_CornerDetect.py`, the parameters are as follows:

```python
    python train_CornerDetect.py \
        -e [epoch] \
        -b [batchsize] \
        -l [learning rate]
```

*Distortion Correction:* You can use the `train_DistCorr.py`, the parameters are as follows:

```python
    python train_DistCorr.py \
        -e [epoch] \
        -b [batchsize] \
        -l [learning rate] \
        -o [parameter order]
```

### Data Generation
You can run `data_generator.py` to generate synthetic dataset with chessboard images and camera parameters.

See `dataset/README.md` for details.
