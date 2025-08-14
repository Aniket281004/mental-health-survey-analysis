from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,classification_report, roc_auc_score, roc_curve,auc,confusion_matrix, ConfusionMatrixDisplay,silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
df=pd.read_csv("clean._data.csv")
features = ['Age','Gender', 'self_employed', 'family_history', 'work_interfere', 'remote_work', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview' ]
X=df[features]
y=df['treatment']
y=y.map({"Yes":1,"No":0})
def build_pipeline(X):
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_trans=Pipeline([
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',StandardScaler())
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     
    random_state=42
    )
pipe=Pipeline([
    ('preprocessor',build_pipeline(X_train)),
    ('clf',LogisticRegression(max_iter=3000))
])
param_grid={
    'clf__C':[0.1,],
    'clf__solver':['saga'],
    'clf__penalty':['l1','l2','elasticnet','None']
    
    
}
gs=GridSearchCV(pipe,param_grid=param_grid,cv=10)
gs.fit(X_train,y_train)
y_prob=gs.predict_proba(X_test)[:,1]
y_pred= (y_prob >= 0.52).astype(int)
print('Best params:',gs.best_params_)
print("Classification Report:",classification_report(y_test,y_pred))
print("ROC-AUC:",roc_auc_score(y_test,y_prob))

