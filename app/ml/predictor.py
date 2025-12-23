from app.ml.loader import load_model
from app.ml.ensemble import ensemble_predict
from app.ml.preprocessing import preprocess_input

xgb = load_model("xgb")
lgb = load_model("lgb")
cat = load_model("cat")

def predict_churn(raw_df):
    X = preprocess_input(raw_df)
    return ensemble_predict(xgb, lgb, cat, X)
