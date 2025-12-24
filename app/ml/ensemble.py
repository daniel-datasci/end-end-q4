def ensemble_predict(xgb, lgb, cat, X):
    return (
        0.4 * xgb.predict_proba(X)[:, 1] +
        0.3 * lgb.predict_proba(X)[:, 1] +
        0.3 * cat.predict_proba(X)[:, 1]
    )
