## Intermediate Machine Learning Project -> Customer Churn Prediction

In this project I implemented the Customer churn prediction on Telecome's customer churn dataset from Kaggle.

First of all, I build the model in google colab. This is the usual process in which I uploaded the dataset then, preprocessed the data
and then applied model training using Radom Forest Classifier. Then I saved the file using joblib to pickle extension and named it "churn_model.pkl".

After that, I made Flask API for the model so that it can now predict on the requested APIs. Then I tested the request data using Postman. 

![image](https://github.com/user-attachments/assets/cfe08373-9906-462c-afd7-5315609a2235)


![image](https://github.com/user-attachments/assets/fe62239e-d126-4d50-8660-96ab7ee9e7f1)
