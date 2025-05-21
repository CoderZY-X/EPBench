<div align="center">

<h1>EPBench: A Benchmark for Short-term Earthquake
Prediction with Neural Networks </h1>

Zhiyu Xu,
Qingliang Chen

</div>

## 🚀 Overview
<div align="center">
<img width="660" height="400" alt="image" src="figs/distribute.png">
</div>

## 📖 Description

We have built a new benchmark (**EPBench**), which is, to our knowledge, the first global regional-scale short-term earthquake prediction benchmark. Our benchmark comprises 924,472 earthquake records and 2959 multimodal earthquake records collected from seismic networks around the world. Each record includes basic information such as time, longitude and latitude, magnitude, while each multimodal record includes waveform and moment tensor information additionally, covering a time span from 1970 to 2021. To evaluate  performance of models on this task, we have established a series of data partitions and evaluation methods tailored to the short-term earthquake prediction task. We also provide a variety of tools to assist future researchers in partitioning the data according to their  geographical understanding. Our benchmark includes a variety of neural network models widely used for time series forecasting, as well as a statistical-based model currently employed by seismological bureaus in several countries. We hope this benchmark will serve as a guide to attract more researchers to address this task.


## 🏗️ Usage

### Dependencies

Some core dependencies:

- pandas==2.2.3
- torch == 1.11.0
- numpy==2.2.6

More details can be found in <./requirements.txt>

### Datasets

More details can be found at:
- EarthQuake1970~2021 Datasets: <https://www.kaggle.com/datasets/drloser/earthquake19702021>
  

### Training

Please first modify the input(training) and output(prediction) file paths in the target model's Python script. All paths are configured within the ```if __name__ == '__main__':``` block.

Then you can run:

```shell
$ python cnn/lstm/transformer/diffusion.py

```
### Testing
Please first modify the input(ground truth and prediction) file paths in the Python script. All paths are configured at the top of the program.

Then for neural network models, you can run:

```shell
$ python evaluation.py
```
For the ETAS models, you can run:

```shell
$ python evaluation-etas.py
```
### ETAS
The details and code of the ETAS model can be found at "https://zenodo.org/records/7584575".
