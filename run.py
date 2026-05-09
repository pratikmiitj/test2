from utils import *

X_train, X_test, y_train, y_test = load_data()
pred,clf = model_fit_predict(X_train, X_test, y_train)

performance(X_test, y_test, pred,clf)