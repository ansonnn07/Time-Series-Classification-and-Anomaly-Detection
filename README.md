# Time Series Classification, Anomaly Detection & Forecasting with PyTorch (ft. PyTorch Lightning)

<div style="display: flex; justify-content: center; align-items: center">

<a href="https://pytorch.org/"><img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/pytorch_logo_icon_169823.png" width="400px"></a>

<a href="https://www.pytorchlightning.ai/"><img src="https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_static/images/logo.png?raw=true" width="400px"></a>

</div>

## Notebooks

**1. Labeling.ipynb**:<br>
This notebook shows how to label time series data using Label Studio for two tasks: classification and anomaly detection. Get the Kaggle Dataset used in this demo from [here](https://www.kaggle.com/c/career-con-2019/data). The dataset is the Career-Con 2019: Help Robots Navigate dataset, which is also used in the second notebook here.

**2. Training - Time Series Classification**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lcEF_rHSl1oS4pZg2NKSLZBTc0O7RpAK?usp=sharing) <br>
This notebook is used to train a time series classification model on the [Kaggle dataset](https://www.kaggle.com/c/career-con-2019/data) to help robots navigate. The latest updates should be referred in Google Colab as the model was trained there.

**3. Training - Time Series Anomaly Detection (PyTorch Lightning).ipynb**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15j6gevN8zGsvo1-uKq2Gc1IB8D-0G0X8?usp=sharing) <br>
This notebook is used to train a time series classification model on the ECG5000 heartbeat dataset obtained from [here](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000), to detect whether there is anomaly in a heartbeat sequence. The latest updates should also be referred in Google Colab as the model was trained there. This notebook makes use of PyTorch Lightning, which is a wrapper of PyTorch framework, to make the training code much easier to implement.

**4. Training - Time Series Forecasting (PyTorch Lightning).ipynb**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zg1JxfpI5RHr89GnKNbhrTZXV_nQa6rz?usp=sharing) <br>
This notebook shows how to forecast time series data. The example dataset used here is the Facebook stock price. Please refer to the Google Colab notebook for latest updates. Note that this is not a full-fledged forecaster because such data are very volatile and very difficult to forecast accurately.

## PyTorch Installation
To install PyTorch with GPU support, refer to the official page from PyTorch [here](https://pytorch.org/get-started/locally/). Or just run the code below in your virtual environment.

```
conda install pytorch torchvision torchaudio cudatoolkit=<YOUR_CUDA_VERSION> -c pytorch -c conda-forge
```

Keep in mind that you will need a compatible CUDA version (preferably v10.2) to work with PyTorch GPU. You can check whether you have installed CUDA or not by checking the existence of any folder inside `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"` (for Windows, I am not sure about Linux), you should see a folder named with CUDA version, e.g. `v10.2` for CUDA version 10.2. You may download and install CUDA from [here](https://developer.nvidia.com/cuda-downloads) if you haven't done so yet. 

NOTE: If you have used TensorFlow with GPU support before, then you have already installed CUDA and cuDNN, if not, you may refer to how to install them in this YouTube [video](https://youtu.be/hHWkvEcDBO0) (without installing the TensorFlow in the same environment of course, since we are using PyTorch here).

If you get an error as shown below:
```
ImportError: DLL load failed while importing win32api: The specified module could not be found.
```
Then you need to run the following commands below to uninstall and reinstall `pywin32`, the uninstall command might tell you that `pywin32` is not found but it's ok, just proceed to the next command.
```
pip uninstall pywin32
conda uninstall pywin32
conda install pywin32
```

## PyTorch Lightning Installation
```
pip install pytorch-lightning
```

## Package Installation
Just run the following command to install the rest of the dependencies:

```
pip install -r requirements.txt
```