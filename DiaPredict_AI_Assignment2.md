# DiaPredict AI: A Supervised Machine Learning Model for Predicting Diabetes Status Using Patient Medical Attributes

### Project Developers:
1. Allan Cheruiyot (SALLCH2311)
2. Isokat Lyne (SISOLY2311)
3. Kelvin Kipruto (SKIPKE2312)

### Results and Discussion

The DiaPredict AI model was successfully developed, trained, and evaluated using four different supervised machine learning algorithms: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Support Vector Machine (SVM). The models were trained on 80% of the dataset and tested on the remaining 20% to evaluate their predictive capabilities. Among the models evaluated, the **Random Forest Classifier** achieved the highest overall performance on the test set, recording an Accuracy of **77.92%**, a Precision of **71.74%**, a Recall (Sensitivity) of **61.11%**, and an F1-Score of **66.00%**. This indicates that the ensemble learning approach of Random Forest was the most effective in capturing the complex, non-linear relationships present in the medical data. The Decision Tree model also performed well with an accuracy of 75.97% and the highest recall rate (72.22%), meaning it was the best at identifying actual positive diabetic cases.

To ensure the models' stability and generalization capabilities, a 5-fold cross-validation was performed on the training data. The cross-validation scores confirmed the robustness of the models, with Logistic Regression averaging 78.18% accuracy, Random Forest averaging 76.88%, and SVM averaging 76.88%. Based on the performance metrics obtained during the testing phase, the Random Forest model was selected as the final predictive engine for the DiaPredict AI system. The model has successfully met its required objective by demonstrating that it can effectively and reasonably predict the likelihood of a patient having diabetes using historical medical attributes, which aligns with the goal of assisting healthcare professionals in early disease screening. 

### Summary and Conclusion

In conclusion, the DiaPredict AI project successfully demonstrates how supervised machine learning algorithms can be leveraged to address the critical public health challenge of delayed diabetes diagnosis. By analyzing key indicators such as glucose levels, BMI, and insulin data from the Pima Indians Diabetes Dataset, we engineered a reliable predictive model driven by the Random Forest algorithm. This data-driven approach offers a proactive solution that can assist medical practitioners in identifying high-risk individuals before severe complications arise. For future improvements and recommendations, the model's functionality and accuracy could be significantly enhanced by expanding the dataset to include a larger, more demographically diverse patient population, which would reduce inherent feature bias. Additionally, integrating the predictive model into a user-friendly mobile or web application could facilitate real-time risk assessment, allowing for wider accessibility by patients and healthcare providers in resource-limited settings.
