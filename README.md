# Custom Time Series Classification & Anomaly Detection

Link to Kaggle Dataset - [CareerCon 2019 - Help Navigate Robots](https://www.kaggle.com/c/career-con-2019/data).

## Notebooks

`1. Labeling.ipynb`: This notebook shows how to label time series data using Label Studio for two tasks: classification and anomaly detection.

`2. Training - Time Series Classification`: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lcEF_rHSl1oS4pZg2NKSLZBTc0O7RpAK?usp=sharing) <br>
This notebook is used to train a time series classification model on the Kaggle dataset to help robots navigate. The latest updates should be referred in Google Colab as the model was trained there. 

`3. Training - Time Series Anomaly Detection (PyTorch Lightning).ipynb`: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15j6gevN8zGsvo1-uKq2Gc1IB8D-0G0X8?usp=sharing) <br>
This notebook is used to train a time series classification model on the ECG5000 heartbeat dataset obtained from [here](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000), to detect whether there is anomaly in a heartbeat sequence. The latest updates should also be referred in Google Colab as the model was trained there. This notebook makes use of PyTorch Lightning, which is a wrapper of PyTorch framework, to make the training code much easier to implement. 



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