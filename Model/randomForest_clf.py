import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,classification_report, roc_auc_score, roc_curve,auc,confusion_matrix, ConfusionMatrixDisplay,silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("clean._data.csv")
features = ['Age','Gender', 'self_employed', 'family_history', 'work_interfere', 'remote_work', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview' ]
X=df[features]
y=df['treatment']
y=y.map({"Yes":1,"No":0})
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     
    random_state=42
    )
from sklearn.ensemble import RandomForestClassifier
def build_processing_pipeline(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_trans=Pipeline([
        ('imputer',SimpleImputer(strategy='mean'))
    ])
    cat_trans=Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        (
            'onehot',OneHotEncoder(handle_unknown='ignore')
        )
    ])
    preprocessor=ColumnTransformer([
        ('nums',num_trans,num_cols),
        ('cat',cat_trans,cat_cols)
    ])
    return preprocessor
pipe_rf=Pipeline([
    ('preprocessor',build_processing_pipeline(X_train)),
    ('clf',RandomForestClassifier(random_state=42))
])
param_grid={
   'clf__n_estimators':[100,200,300],
   'clf__max_depth':[None,10,20],
   'clf__min_samples_split':[2,5],
   'clf__min_samples_leaf':[1,2]
}
gs_rf=GridSearchCV(pipe_rf,param_grid=param_grid,cv=5,n_jobs=-1)
gs_rf.fit(X_train,y_train)
y_prob_rf=gs_rf.predict_proba(X_test)[:,1]
y_pred_rf = (y_prob_rf >= 0.524).astype(int)
print('Best params:',gs_rf.best_params_)
print("Classification Report:",classification_report(y_test,y_pred_rf))
print("ROC-AUC:",roc_auc_score(y_test,y_prob_rf))


