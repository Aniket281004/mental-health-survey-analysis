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
df=pd.read_csv(clean_data.csv)
features = ['Gender', 'self_employed', 'family_history', 'treatment', 'work_interfere', 'remote_work', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview' ]
X = df[features]
y = df['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)
preprocessor2 = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)
models = {
    
    "LinearRegression": (
        LinearRegression(),
        {
            'model__fit_intercept': [True, False],
            'model__positive': [False, True]
        }
    )
}
for name, (estimator, params) in models.items():
    if estimator=='RandomForestRegressor':
        pipe = Pipeline([
        ('preprocessor', preprocessor2),
        ('model', estimator)
        ])
    else:
        pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', estimator)
        ])

    
    grid = GridSearchCV(pipe, params, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"\n🔹 {name} Results:")
    print(f"Best CV R²: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")
    
    y_pred = grid.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    mape = (abs((y_test - y_pred) / y_test).mean())*100
    
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²:   {r2:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
