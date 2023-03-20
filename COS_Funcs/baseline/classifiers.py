from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np

def get_dt(X_train, y_train):
    param_grid = { 'criterion':['gini','entropy'],
                  'max_depth': np.arange(3, 200),
                  'max_features': ['auto', 'log2'],
                  }
    model=DecisionTreeClassifier()
    adb = GridSearchCV(model, param_grid, cv=5,scoring='f1_weighted')
    adb.fit(X_train, y_train)
    return adb.best_estimator_ 

def do_classification(X_train,y_train,X_test,classification_model):
    
    if classification_model == 'knn':
        model = KNeighborsClassifier()
    
    elif classification_model == 'svm':
        model = SVC()
        
    elif classification_model == 'decision_tree':
        model = DecisionTreeClassifier()
    
    elif classification_model == 'random_forest':
        model = RandomForestClassifier()
    
    elif classification_model == 'mlp':
        model = MLPClassifier()
    
    elif classification_model == 'naive_bayes':
        model = GaussianNB()
    
    model.fit(X_train,y_train)
    return model.predict(X_test)