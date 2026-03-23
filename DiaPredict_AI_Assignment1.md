# DiaPredict AI: A Supervised Machine Learning Model for Predicting Diabetes Status Using Patient Medical Attributes

### Project Developers:
1. Allan Cheruiyot (SALLCH2311)
2. Isokat Lyne (SISOLY2311)
3. Kelvin Kipruto (SKIPKE2312)

## 1. Background of the Problem / Problem Statement

Artificial Intelligence (AI) has increasingly transformed healthcare by enabling predictive analytics and intelligent decision-making systems. One of the most significant global public health challenges today is diabetes mellitus. According to the World Health Organization (WHO, 2023), diabetes is a chronic metabolic disease characterized by elevated blood glucose levels, which over time leads to serious damage to the heart, blood vessels, eyes, kidneys, and nerves. The global prevalence of diabetes has been rising steadily, particularly in low- and middle-income countries, placing pressure on already strained healthcare systems.

In Kenya and many developing nations, early diagnosis of diabetes remains a major challenge due to limited screening facilities, inadequate healthcare infrastructure, and lack of awareness among the population (WHO, 2023). Many individuals are diagnosed only after complications such as kidney failure, cardiovascular disease, or vision impairment have already developed. Traditional diagnostic approaches rely on laboratory blood tests and physician evaluation. While these methods are medically reliable, they are often reactive rather than preventive. Patients typically undergo testing only after symptoms appear, which reduces the opportunity for early intervention (American Diabetes Association, 2022).

Various strategies have been implemented to address this issue, including public health awareness campaigns, routine hospital screenings, and manual risk assessment tools. However, these approaches depend heavily on human intervention and may not effectively identify high-risk individuals before complications occur. With the advancement of Artificial Intelligence and Machine Learning, predictive models can analyze historical medical data to detect patterns associated with diabetes risk. The availability of structured datasets such as the Pima Indians Diabetes Dataset from the UCI Machine Learning Repository and Kaggle makes it possible for our group to develop accurate supervised learning models for early disease prediction (UCI Machine Learning Repository, n.d.; Kaggle, n.d.). Therefore, developing DiaPredict AI offers a proactive, data-driven solution capable of assisting healthcare professionals in identifying high-risk individuals early, enabling timely intervention, reducing complications, and lowering long-term treatment costs.

## 2. Methodology

### i. Objective of the Study
The primary objective of this project is to design and implement a supervised machine learning model capable of predicting the likelihood of diabetes in patients based on medical attributes. We aim to analyze patient health indicators and classify individuals into either diabetic or non-diabetic categories. By doing so, our model supports healthcare professionals in early screening and preventive care strategies, aligning with global recommendations for early diabetes detection (WHO, 2023; American Diabetes Association, 2022).

### ii. Steps in Creating the Machine Learning Model
The development of DiaPredict AI will follow a structured machine learning lifecycle to ensure accuracy and reliability. Our group will follow the steps outlined below:
1. **Problem Definition** – We will formulate diabetes prediction as a binary classification problem where the output variable has two possible values: Diabetic (1) or Non-Diabetic (0).
2. **Data Collection** – We will obtain the dataset from the UCI Machine Learning Repository and Kaggle (UCI Machine Learning Repository, n.d.; Kaggle, n.d.).
3. **Data Understanding** – We will conduct Exploratory Data Analysis (EDA) to examine the distribution of features such as glucose level, BMI, age, and insulin levels. Statistical summaries and visualizations (e.g., histograms and correlation matrices) will be used to understand relationships between variables.
4. **Data Preprocessing** – We will clean and prepare the dataset to remove inconsistencies and handle missing values.
5. **Feature Selection and Engineering** – We will identify the most influential medical indicators contributing to diabetes prediction.
6. **Model Training** – We will apply supervised classification algorithms to train predictive models.
7. **Model Evaluation and Validation** – We will measure model performance using statistical evaluation metrics.
8. **Model Selection** – We will choose the most accurate and balanced model for final deployment.

### iii. Dataset Source and Size
The dataset used in this project is the Pima Indians Diabetes Dataset available from the UCI Machine Learning Repository and Kaggle. Our group will use a dataset containing:
- 768 patient records
- 8 independent medical attributes
- 1 binary outcome variable

The features include:
- Number of Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age

### iv. Data Preprocessing and Cleaning
Data preprocessing is critical to ensure model reliability. We will use the following steps to implement data preprocessing and cleaning:
- **Handling Missing Values:** Some attributes such as glucose, blood pressure, and BMI may contain zero values that are medically unrealistic. We will replace these values using statistical imputation methods such as mean or median substitution.
- **Outlier Detection:** Extreme values will be identified and handled using boxplots and statistical thresholds to prevent distortion of model training.
- **Feature Scaling:** Standardization (e.g., StandardScaler) will be applied to normalize numerical values. This is especially important for algorithms like Support Vector Machine, which are sensitive to feature magnitude differences.
- **Data Splitting:** We will divide the dataset into 80% training data and 20% testing data to evaluate generalization performance.
- **Cross-Validation:** We will apply k-fold cross-validation (k=5 or 10) to ensure model stability and reduce overfitting.

### v. Supervised Learning Algorithms to be Used
We will implement the following supervised classification algorithms:
- **Logistic Regression** – Serves as a baseline model due to its interpretability and effectiveness in binary classification problems.
- **Decision Tree Classifier** – Captures nonlinear relationships between features.
- **Random Forest Classifier** – An ensemble learning method that improves accuracy by combining multiple decision trees.
- **Support Vector Machine (SVM)** – Effective in high-dimensional feature spaces and capable of handling classification boundaries efficiently.

Each model will be trained using the labeled dataset and compared based on performance metrics.

### vi. Model Validation and Evaluation
To ensure robust performance, We will use the following evaluation metrics:
- **Accuracy** – Measures overall correctness of predictions.
- **Precision** – Measures how many predicted diabetic cases are actually diabetic.
- **Recall (Sensitivity)** – Measures how well the model identifies actual diabetic patients (critical in healthcare).
- **F1-Score** – Balances precision and recall.
- **Confusion Matrix** – Provides detailed insight into true positives, false positives, true negatives, and false negatives.

Cross-validation will be applied to ensure the model generalizes well beyond the training dataset. The model with the best balance between sensitivity and overall accuracy will be selected for deployment.

## Conclusion
DiaPredict AI demonstrates the practical application of supervised machine learning in solving a real societal problem in healthcare. By leveraging historical patient data from reputable repositories such as the UCI Machine Learning Repository and Kaggle, our group has designed an intelligent predictive tool for early diabetes detection. In alignment with global health recommendations (WHO, 2023; American Diabetes Association, 2022), this AI-based approach enables proactive risk identification and supports preventive healthcare strategies. The project is feasible, data-driven, cost-effective, and academically appropriate for supervised machine learning implementation.

### References
- World Health Organization. (2023). Diabetes fact sheet.
- American Diabetes Association. (2022). Classification and diagnosis of diabetes: Standards of medical care in diabetes.
- UCI Machine Learning Repository. (n.d.). Pima Indians Diabetes Dataset.
- Kaggle. (n.d.). Pima Indians Diabetes Database.
