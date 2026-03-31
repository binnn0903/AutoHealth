#### 1) **Deep Ensemble**

**Estimation Type:** ⭐⭐⭐⭐ epistemic + some aleatoric  
**How to use:** Train multiple models with the same architecture but different random seeds / data sampling, and measure disagreement among their predictions.  
**Advantages:**

- Widely recognized as the strongest and most stable baseline
- More sensitive to OOD samples and hard examples
  **Disadvantages:**
- High training and storage cost (N times larger)
  **Suitable for:** High-risk tasks (medical, finance) and scenarios requiring robust confidence estimates.

------

#### 2) **Temperature Scaling**

**Estimation Type:** ⭐⭐⭐ makes probabilities more "trustworthy" (calibration)  
**How to use:** Learn a temperature parameter `T` on a validation set and use it to adjust the softmax outputs.  
**Advantages:**

- Extremely simple and almost always worth doing
- Does not change accuracy but can significantly improve reliability
  **Disadvantages:**
- Cannot identify epistemic uncertainty or OOD; it only "corrects probabilities"
  **Suitable for:** Any classification setting where the model outputs probabilities.

------

#### 3) **MC Dropout**

**Estimation Type:** ⭐⭐⭐ epistemic (approximate Bayesian)  
**How to use:** Keep dropout enabled during inference, run `T` forward passes, and compute the mean and variance.  
**Advantages:**

- One of the easiest epistemic uncertainty methods to implement in practice
- Works with a single model
  **Disadvantages:**
- The quality of uncertainty estimates depends on dropout design
- Often weaker than ensemble methods
  **Suitable for:** Cases where you want a quick epistemic method under limited compute budget.

------

#### 4) **Heteroscedastic Regression (Output Mean + Variance)**

**Estimation Type:** ⭐⭐⭐⭐ aleatoric (data noise)  
**How to use:** Let the model output `\mu(x)` and `\sigma(x)`, and train it with NLL.  
**Advantages:**

- The most practical standard solution for regression tasks
- Can learn different noise levels for different samples
  **Disadvantages:**
- Weak at capturing epistemic uncertainty unless combined with ensemble methods or Bayesian approaches
  **Suitable for:** Tasks where prediction error depends on the input, such as perception, time series, and price forecasting.

------

#### 5) **Quantile Regression / Prediction Intervals**

**Estimation Type:** ⭐⭐⭐ aleatoric + prediction intervals  
**How to use:** Directly train the model to predict quantiles such as `q0.1`, `q0.5`, and `q0.9`.  
**Advantages:**

- Directly outputs confidence intervals, which is very practical
- No need to assume Gaussian noise
  **Disadvantages:**
- Requires training multiple quantiles
  **Suitable for:** Risk-sensitive prediction, confidence intervals, risk control, and demand forecasting.

------

#### 6) **Bayesian Neural Networks (VI / Laplace Approx.)**

**Estimation Type:** ⭐⭐⭐⭐ epistemic (theoretically principled)  
**How to use:** Use VI or Laplace approximation to learn the posterior over parameters, then average predictions over the posterior at inference time.  
**Advantages:**

- The most theoretically rigorous approach
- Performs well in small-data or out-of-distribution settings
  **Disadvantages:**
- Engineering complexity is high and computation is expensive
- Performance varies significantly across approximation methods
  **Suitable for:** Small-data, high-risk scenarios, or settings that require theoretical guarantees.

------

#### 7) **SWAG (Stochastic Weight Averaging Gaussian)**

**Estimation Type:** ⭐⭐⭐ epistemic (approximate posterior)  
**How to use:** Use SWA to track the mean and covariance of model weights, then sample weights to obtain uncertainty estimates.  
**Advantages:**

- Sits between ensemble methods and BNNs
- Cheaper than deep ensembles
  **Disadvantages:**
- Sensitive to training strategy
  **Suitable for:** Cases requiring relatively strong epistemic uncertainty estimates but without enough compute budget for ensembles.

------

#### 8) **Snapshot Ensemble**

**Estimation Type:** ⭐⭐⭐ epistemic  
**How to use:** Use cyclical learning rates such as cosine annealing, save multiple checkpoints, and ensemble them.  
**Advantages:**

- An ensemble with "almost no extra training cost"
- More stable than MC Dropout
  **Disadvantages:**
- Less diversity than truly independent models
  **Suitable for:** Deep learning tasks where training is expensive but ensemble benefits are still desired.

------

#### 9) **Evidential Deep Learning (Dirichlet-Based)**

**Estimation Type:** ⭐⭐⭐ epistemic-like + some OOD tendency  
**How to use:** Output Dirichlet parameters (evidence); less evidence indicates higher uncertainty.  
**Advantages:**

- A single model can provide distribution-level uncertainty
- Has some effectiveness for OOD detection
  **Disadvantages:**
- Sensitive to loss design and may produce "incorrect confidence"
- Real-world performance is often weaker than ensembles
  **Suitable for:** Cases where a single model must output uncertainty and some OOD capability is needed.

------

#### 10) **Energy-Based / OOD Scoring**

**Estimation Type:** ⭐⭐⭐⭐ OOD (out-of-distribution detection)  
**How to use:** Use an energy score or ODIN-like method to assign a score indicating whether a sample resembles the training distribution.  
**Advantages:**

- Often stronger than plain softmax for OOD detection
- Can be added on top of any classifier
  **Disadvantages:**
- Does not necessarily provide uncertainty in the strict probabilistic sense
  **Suitable for:** Open-set recognition and deployment safety, where unknown inputs should be rejected.
