import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.data
# mlflow.autolog()
# mlflow.set_experiment("water_auto")
# mlflow.set_tracking_uri("http://127.0.0.1:5000")



mlflow.set_experiment("watertest_randomizedsearchcv")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data= pd.read_csv(r"C:\Users\u350272\Desktop\projectdata\e2eflowdata\water_potability.csv")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
train_data, test_data = train_test_split(data,test_size=0.2,random_state=42)

def fillmissingvalues(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)

    return df


train_processed_data = fillmissingvalues(train_data)
test_processed_data= fillmissingvalues(test_data)



from sklearn.ensemble import RandomForestClassifier
import pickle

X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values


rf = RandomForestClassifier(random_state=42)

param_dist ={
    'n_estimators':[100,200,300,400,500],
    'max_depth':[None,10,20,30,40],

}

random_search=RandomizedSearchCV(estimator=rf,param_distributions=param_dist,n_iter=50,cv=5,n_jobs=-1, verbose=2, random_state=42)

# n_estimator= 1000
with mlflow.start_run(run_name="Random forest tuning") as parent_run:
    # cls= RandomForestClassifier (n_estimators=n_estimator)
    random_search.fit(X_train,y_train)

    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_param(f"parameters for {i+1}",random_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])

    print("Best Params: ", random_search.best_params_)

    best_rf = random_search.best_estimator_

    best_rf.fit(X_train,y_train)
    pickle.dump(best_rf,open("model.pkl","wb"))


    X_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values



    y_pred = best_rf.predict(X_test)


    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    accuracyscore = accuracy_score(y_test,y_pred)
    precisionscore = precision_score(y_test,y_pred)
    recallscore = recall_score(y_test,y_pred)
    f1score = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("confusion matrix")

    plt.savefig("confusion_matrix.png")


    mlflow.log_metric("accuracy",accuracyscore)
    mlflow.log_metric("precision",precisionscore)
    mlflow.log_metric("recall",recallscore)
    mlflow.log_metric("f1score",f1score)

    mlflow.log_param("best parameters",random_search.best_params_)

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(random_search.best_estimator_,"Randomforest classifier")

    mlflow.log_artifact(__file__)

    training_data = mlflow.data.from_pandas(train_processed_data)
    test_data = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(training_data,"training_data")
    mlflow.log_input(test_data,"test _data")


    print("accuracy score:",accuracyscore)
    print("precision score:",precisionscore)
    print("recall score:",recallscore)
    print("f1 score:",f1score)
