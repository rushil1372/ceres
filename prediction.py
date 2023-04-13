import joblib

def predict(data):
    clf = joblib.load("abc.h5")
    return clf.predict(data)