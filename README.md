# AGSTN: Forecasting Urban Sensory Values through Learning Attention-adjusted Graph Spatio-Temporal Networks
Official implementation of AGSTN model(ICDM2020)
* Abstract:

Forecasting spatio-temporal correlated time series of sensor values is crucial in urban applications, such as air pollution alert, biking resource management, and intelligent transportation systems. While recent advances exploit graph neural networks (GNN) to better learn spatial and temporal dependencies between sensors, they cannot model time-evolving spatio-temporal correlation (STC) between sensors, and require pre-defined graphs, which are neither always available nor totally reliable, and target at only a specific type of sensor data at one time. Moreover, since the form of time-series fluctuation is varied across sensors, a model needs to learn fluctuation modulation. To tackle these issues, in this work, we propose a novel GNN-based model, \textit{Attention-adjusted Graph Spatio-Temporal Network} (AGSTN). In AGSTN, multi-graph convolution with sequential learning is developed to learn time-evolving STC. Fluctuation modulation is realized by a proposed attention adjustment mechanism. Experiments on three sensor data, air quality, bike demand, and traffic flow, exhibit that AGSTN outperforms the state-of-the-art methods.

<img src="https://github.com/l852888/AGSTN/blob/master/overview.PNG" width="75%" height="75%">

We propose a novel model,Attention-adjusted Graph Spatio-Temporal Network (AGSTN), to deal with the SSVF problem.
AGSTN consists five parts. 
* First, we exploit Intrinsic Mode Functions (IMF) to generate additional features from the original time series. 
* Second, we aim at learning spatiotemporal correlation between sensors based on constructed graphs. Multi-graph convolutional network (M-GCN) layers are incorporated to achieve the aim. 
* Third, we use the sequential methods to make raw predictions from the original time series and the M-GCN-generated features, using recurrent and convolutional neural networks, respectively, from which time-evolving spatio-temporal correlation between sensors is modeled. 
* Fourth, we impose an attention layer to learn the reasonable fluctuation tendency of every sensor. Last, by averaging raw predictions and applying the attention adjustment,
the final prediction is generated.

Datasets
------------------
Three datasets of spatio-temporal correlated time series are employed in our experiments. 
The types of timeseries values include air quality, bike demand, and traffic flow.
* air quality: we collect hourly PM2.5 data of northern region from Environmental Protection Agency in Taiwan.
* bike demand: we make use of the Citi Bike public dataset in New York City, whose sensors collect the demand number of bikes every 30 minutes.
* traffic flow: we utilize the METR-LA dataset, which contains traffic information collected from loop detectors in the highway of Los Angeles. The sensor readings are aggregated into 5-minutes windows.

Requirements
------------------
python >=3.5

Keras 2.2.4

scikit-learn 0.21

pandas 0.23.0

numpy 1.14.3

Citation
------------------------
Yi-Ju Lu and Cheng-Te Li. "AGSTN: Forecasting Urban Sensory Values through Learning Attention-adjusted Graph Spatio-Temporal Networks" 20th IEEE International Conference on Data Mining, ICDM2020.
