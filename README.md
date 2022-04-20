## Time Series Forecasting With Deep Learning

### Problem Statement

Quantifying and predicting the movement of publicly traded securities is a lucrative challenge that has pushed the limits of deep learning since its first applications to the field decades ago.

At its simplest, this process can be (incorrectly) abstracted as a martingale process, where the conditional expectation of the next value in a sequence is equal to the present value, regardless of all previous performance. Specifically, we can say that

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?E(X_{n&plus;1}|X_1,...,X_n&space;)&space;=&space;X_n">
</p>

An example of an “ideal” martingale would be an unbiased random walk in any number of dimensions. Modeling such a process would be relatively simple, even without full knowledge of the random variable. Given a function g(x) of a random variable x, if we know the probability distribution of x but not the distribution of g(x), we can make a reasonable expectation

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?(i):E[g(x)]=\sum_{x}g(x)f_X(x)">
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?(ii):E[g(x)]=\int_{-\infty}^{\infty}g(x)f_X(x)dx">
</p>

for discrete (i) and continuous (ii) processes, where f_x (x) is the probability mass function. This theorem, called Law of the Unconscious Statistician, is one example of a postulate from measure theory that can be used and proved to model these processes.

However, if something like the S&P-500 index traded like a true martingale, a simple multivariate Kalman filter would be roughly sufficient to model it. By simple experimentation it is apparent this is not true. To truly appreciate the underlying relationships, the stationarity of the process itself must be quantified. By definition, a stationary process is one where its unconditional joint probability distribution will shift in time; consequently, metrics such as mean and variance will also change over time.

Non-stationarity can be confirmed with tests such as the augmented Dickey-Fuller, which test data sets against a null hypothesis that a unit root is present in the characteristic equation. Failing to reject the null hypothesis indicates that the sample is stationary, and contains at least one unit root.


### Tasks, Challenges, Methodology

Many modeling techniques exist for the quantification and prediction of financial price movement, and all are grounded in the fundamentals of time-series analysis. Calibrating these models can be seen as an inverse to how the problem is traditionally approached: given observed outputs, the input parameters must be inferred (Liu, 2019).

Time-series analysis is limited by the pre-assumption of the linear form of the model. Specifically, linear correlation is assumed among the time-series values. It is in this space that deep learning can provide a potential advantage to pre-existing models.

The current SDE-based models, such as ARMA and ARIMA, function by combining the auto-regressive and moving-average model into

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?X_t=c&plus;\varepsilon_t&plus;\sum_{i=1}^{p}\varphi&space;_iX_{t-1}&plus;\sum_{i=1}^{p}\varepsilon_{t-1}">
</p>

As we can see, they function off of some knowledge of previous state, summed as a product with φ_i and θ_i, which are the parameters of the model.

Recurrent neural networks (RNNs) can be used to generalize autoregressive models. RNNs have a mechanism for persistent memory, plus the advantage of flexible non-linear modeling capability (Dixon, 2021). This opens the door for two different hybrid approaches. First is a traditional model calibrated by an RNN, while the other is an RNN that fully integrates the autoregressive and moving average SDE’s into its recurrent architecture.


### Dataset

For this project, one dataset will be modeled based on its usability and completeness -- a 1-minute bitcoin historical price record (Kaggle). It includes open/high/low/close prices for the 1-minute window, along with volume and volume-weighted price. For the purposes of this data analysis, volume-weighted price will be the focus. Future exploration will likely model S&P 500 stocks and index (also on Kaggle).


### Works Referenced

- Calvo-Pardo, H. F. (2020). Neural Network Models for Empirical Finance. Journal of Risk and Financial Management.
- Dixon, M. (2021). Financial Forecasting With a-RNNs: A Time Series Modeling APproach. Frontiers in Applied Mathematics and Statistics.
- Kaggle. (n.d.). Bitcoin Historical Data. Retrieved from kaggle.com: https://www.kaggle.com/mczielinski/bitcoin-historical-data¬¬
- Liu, S. (2019). A Neural Network-Based Framework for Financial Model Calibration. Journal of Mathematics in Industry.
- Revach, G. (2022). KalmanNet: Nueral Network Aided Kalman Filtering for Partially Known Dynamics. IEEE.
- Stubberud, S. (1995). An Adaptive Extended Kalman Filter Using Artificial Neural Networks. Proceedings of the 34th Conference on Decision & Control.
- Zhang, G. P. (1999). Time Series Forecasting Using a Hybrid ARIMA and Neural Network Model. Neurocomputing.
