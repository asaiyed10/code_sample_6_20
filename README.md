

Ahson Saiyed
Date: 5-7-2024

## Overview

This project focuses on the Banana Dataset and implements a Logistic Regression model with various optimizers, including Adam, Newton's Method, Stochastic Gradient Descent, AdaGrad, RMSProp, and a Custom Stochastic Gradient Descent.

**NOTE:** Data Preprocessing has custom modifications which impact testing. Please refer to the `ioutils.py` section below for details.

## Model and Optimizers

- Model: Logistic Regression
- Optimizers:
  - Adam
  - Newton's Method
  - Stochastic Gradient Descent
  - AdaGrad
  - RMSProp
  - Custom Stochastic Gradient Descent

## Usage

### Reproducing Submitted/Fitted Model

To reproduce the model chosen for submission (Logistic Regression + Adam), execute the following command:

```bash
python train.py
```

The fitted model file will be saved asn `new_model.pkl` to prevent `model.pkl` from being overwritten. 

### Running Different Model + Optimizer Pairs

To print results across different Model + Optimizer pairs (e.g., Logistic Regression + Newton's Method, Logistic Regression + Stochastic Gradient Descent, etc.), use the following command:

```bash
python train.py all
```

### Numerical Precision Analysis

To run the numerical precision analysis and reproduce the figures in the report:

```bash
python analysis.py
```

The script, as is, will produce all figures with 'adam', 'rmsprop', and 'float16' for a few iterations. To reproduce the exact figures in the paper, change `max_iter` to 10,000 and run using the default settings.

### Testing the Model

To test the performance of the fitted model, run the following command:

```bash
python test.py
```

## Data Preprocessing

**Note:** If an external testing script is used to test the performance of the model, the feature engineering process described below is necessary. If the testing script in this repository is used, everything will work as expected.

The `ioutils.py` file has been modified to include a method called `readDataPoly(degree=3)`, which generates polynomial features from the original feature set (or any other dataset titled 'banana_quality.csv' in the same directory).

This feature engineering is built into the `train.py` and `test.py` scripts.

## Environment Setup

To set up the environment and run the scripts, follow these steps:

1. Create a virtual environment:
   ```bash
   python -m venv $env_name
   ```

2. Activate the virtual environment:
   ```bash
   source $env_name/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

Note: The only additional packages are `tqdm` and `seaborn`.


