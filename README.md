# DiaPredict AI

A supervised machine learning predictive model developed to predict the likelihood of diabetes in patients based on their medical attributes, specifically using the Pima Indians Diabetes Dataset.

## Project Developers
1. **Allan Cheruiyot** (SALLCH2311)
2. **Isokat Lyne** (SISOLY2311)
3. **Kelvin Kipruto** (SKIPKE2312)

## Overview
This project was developed for the Artificial Intelligence course. The system extracts complex, non-linear insights from medical history to predict a binary outcome: Diabetic (1) or Non-Diabetic (0). The repository contains the complete Machine Learning pipeline evaluating four algorithms: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Support Vector Machine.

## Requirements
* Python 3.8+
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

## Usage
1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the primary ML pipeline script from the root of the project:
   ```bash
   python src/diapredict_model.py
   ```
4. Performance metrics will be printed in the console and visualization charts will be generated in the `outputs/` directory.

## Results
The Random Forest classifier yielded the highest overall test accuracy (77.92%), precision (71.74%), and F1-Score (66.00%). Furthermore, 5-fold cross-validation demonstrated testing stability across subsets of the training data. See the `DiaPredict_AI_Assignment2.md` document for the detailed findings and evaluations regarding the societal implementation and objectives.
