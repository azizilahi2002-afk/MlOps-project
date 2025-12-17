import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


df = pd.read_parquet(r'C:\\Users\\azizi\\mini-mlops-project\\data\\processed\\features_daily.parquet')

X=df[['ca_total','nb_commandes']]
Y=df['label']
split_idx = int(len(df)*0.8)
X_train=X.iloc[:split_idx]
X_test=X.iloc[split_idx:]
Y_train = Y.iloc[:split_idx]
Y_test = Y.iloc[split_idx:]
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\n ...")
model.fit(X_train,Y_train)
print("entrainement terminé")
Y_pred = model.predict(X_test)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred)
r2=r2_score(Y_test,Y_pred)
print("\n=== RÉSULTATS D'ÉVALUATION ===")
print(f"MAE  : {mae:.2f}")    
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.3f}")    


model_path = Path('C:/Users/azizi/mini-mlops-project/models/random_forest_model.pkl')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
print(f"\nModèle sauvegardé: {model_path}")