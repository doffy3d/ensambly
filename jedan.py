import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib

data = np.loadtxt('data_2.csv', delimiter=',')
np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

def column(matrix, p):
    return np.array([row[p] for row in matrix])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Broj stabala
n_estimators = [int(x) for x in np.linspace(start=10, stop=150, num=20)]

#Broj prediktora
max_features = ['sqrt', 'log2', 2, 3, 4, 5, 6]

#Max dubina
max_depth = [int(x) for x in np.linspace(3, 50, num=20)]
max_depth.append(None)


param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}

rf = RandomForestClassifier()
rf_random = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
rf_random.fit(X_train, y_train.ravel())

#rf_random.best_params_
print(rf_random.best_params_)

optimal_max_depth = rf_random.best_params_['max_depth'] #7
optimal_max_features = rf_random.best_params_['max_features'] #4
optimal_n_estimators = rf_random.best_params_['n_estimators'] #54

#print(optimal_max_depth)
#print(optimal_max_features)
#print(optimal_n_estimators)

clf = RandomForestClassifier(max_depth=optimal_max_depth, max_features=optimal_max_features, n_estimators=optimal_n_estimators)
clf.fit(X_train, y_train.ravel())

y_pred = clf.predict(X_test)
print("Tacnost:", metrics.accuracy_score(y_test, y_pred) * 100)

optimal_max_depth_graph = np.linspace(3, 50, 20, dtype=int)

x1_osa = optimal_max_depth_graph
y1_osa = []

for i in range(0, len(optimal_max_depth_graph)):
    clf = RandomForestClassifier(max_depth=optimal_max_depth_graph[i], max_features=optimal_max_features, n_estimators=optimal_n_estimators)

    clf.fit(X_train, y_train.ravel())

    y_pred = clf.predict(X_test)

    print("Tacnost za dubinu", optimal_max_depth_graph[i], "je:", metrics.accuracy_score(y_test, y_pred) * 100)
    y1_osa.append(metrics.accuracy_score(y_test, y_pred) * 100)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
#velicina stabla
ax1.stem(x1_osa, y1_osa)
plt.xlabel('Maksimalna dubina stabla')
plt.ylabel('Tacnost[%]')
plt.xticks(x1_osa)
plt.show()

optimal_max_features_graph = [2, 3, 4, 5, 6]

x2_osa = optimal_max_features_graph
y2_osa = []

for i in range(0, len(optimal_max_features_graph)):
    clf = RandomForestClassifier(max_depth=optimal_max_depth, max_features=optimal_max_features_graph[i], n_estimators=optimal_n_estimators)  # menjati

    clf.fit(X_train, y_train.ravel())

    y_pred = clf.predict(X_test)

    print("Tacnost za maksimalan broj odlika koje se razmatraju", optimal_max_features_graph[i], "je:",
          metrics.accuracy_score(y_test, y_pred) * 100)
    y2_osa.append(metrics.accuracy_score(y_test, y_pred) * 100)

fig2 = plt.figure()
ax2 = fig2.add_subplot()
plt.stem(x2_osa, y2_osa)
plt.xlabel('Maksimalan broj odlika')
plt.ylabel('Tacnost[%]')
plt.show()

optimal_n_estimators_graph = np.linspace(10, 150, 20, dtype=int)

x3_osa = optimal_n_estimators_graph
y3_osa = []

for i in range(0, len(optimal_n_estimators_graph)):
    clf = RandomForestClassifier(max_depth=optimal_max_depth, max_features=optimal_max_features, n_estimators=optimal_n_estimators_graph[i])  # menjati

    clf.fit(X_train, y_train.ravel())

    y_pred = clf.predict(X_test)

    print("Tacnost za broj estimatora", optimal_n_estimators_graph[i], "je:", metrics.accuracy_score(y_test, y_pred) * 100)
    y3_osa.append(metrics.accuracy_score(y_test, y_pred) * 100)

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.stem(x3_osa, y3_osa)
plt.xlabel('Broj stabala')
plt.ylabel('Tacnost[%]')
plt.xticks(x3_osa)
plt.show()