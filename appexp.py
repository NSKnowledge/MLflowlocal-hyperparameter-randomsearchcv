import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
mlflow.set_experiment("water_test_GB")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
data= pd.read_csv(r"C:\Users\u350272\Desktop\projectdata\e2eflowdata\water_potability.csv")


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data,test_size=0.2,random_state=42)

def fillmissingvalues(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)

    return df


train_processed_data = fillmissingvalues(train_data)
test_processed_data= fillmissingvalues(test_data)

train_df = mlflow.data.from_pandas(train_processed_data)
test_df = mlflow.data.from_pandas(test_processed_data)

from sklearn.ensemble import GradientBoostingClassifier
import pickle

X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values


with mlflow.start_run():
    n_estimator= 1000

    cls= GradientBoostingClassifier (n_estimators=n_estimator)
    model = cls.fit(X_train,y_train)


    pickle.dump(cls,open("model.pkl","wb"))


    X_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values



    y_pred = cls.predict(X_test)


    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    accuracyscore = accuracy_score(y_test,y_pred)
    precisionscore = precision_score(y_test,y_pred)
    recallscore = recall_score(y_test,y_pred)
    f1score = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy score",accuracyscore)
    mlflow.log_metric("precision score",precisionscore)
    mlflow.log_metric("recall score",recallscore)
    mlflow.log_metric("f1 score",f1score)

    mlflow.log_param("n_estimator", n_estimator)


    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("confusion matrix")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(cls, "GradientBoostingClassifier")
    mlflow.log_artifact(__file__)

    mlflow.set_tag("author", "NS")
    mlflow.set_tag("Model","GB")

    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df,"test")


    print("accuracy score:",accuracyscore)
    print("precision score:",precisionscore)
    print("recall score:",recallscore)
    print("f1 score:",f1score)
