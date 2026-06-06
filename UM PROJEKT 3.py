import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score

#1. Wczytanie danych

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

dane = pd.read_csv("processed_dataset.csv", sep = ",", decimal = '.')
print('\n Wczytanie danych:')
print(dane.head())

#2. Usunięcie zbędnych kolumn

dane = dane.drop(columns = ["Unnamed: 0","name","blood_pressure"]) #Usuwamy 'blood pressure', ponieważ w danych są już kolumny 'systolic' i 'diastolic', które są podzieloną kolumną 'blood pressure'
print('\n Wczytanie danych zaktualizowanych o usunięcie kolumn: Unnamed: 0, name, blood_pressure:')
print(dane.head())

#3. Modyfikacja zmiennych kategorycznych na numeryczne

#Funkcja get dummies - automatycznie dzieli kolumny z więcej niz 2 zmiennymi, rozdziela je na osobne kolumny i przyporządkowuje im wartości tak albo nie
dane = (pd.get_dummies (dane, columns=['bp_category','obesity_group', ], drop_first=True))

#Zmiana zmiennych tak i nie na zmienne liczbowe 1 i 0
dane_najwyst = dane.astype({col: 'int' for col in dane.select_dtypes('bool').columns})
print('\n Prezentacja typów danych poddanych analizie po zastosowaniu modyfikacji zmiennych na numeryczne:')
print(dane_najwyst.dtypes)

#W tym momęcie mamy przygotowane dane do dalszych obliczeń - usunięte zbędne kolumny, dane zmienione na zmienne numeryczne
print('\n Dane po modyfikacji - zmienne numeryczne:')
print(dane.head())

#4. Podział na X i Y - z pominięciem 'has_eye_disease'

X = dane_najwyst.drop('has_eye_disease', axis = 1)
Y = dane_najwyst['has_eye_disease']

#5. Podział na zbiór uczący i testowy

X_train, X_test, Y_train, Y_test = (train_test_split (X, Y, test_size=0.2, random_state=50))

#6. Standaryzacja

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#7. Optymalizacja hiperparametrów Bagging
param_grid_bagging = { 'n_estimators': [50, 100, 200], 'max_samples': [0.5, 0.7, 1.0], 'max_features': [0.5, 0.7, 1.0],
    'estimator__max_depth': [5, 8, 10]}

bagging = BaggingClassifier(estimator=DecisionTreeClassifier(),random_state=50)

grid_bagging = GridSearchCV(estimator=bagging,param_grid=param_grid_bagging, scoring='accuracy',
    cv=5, n_jobs=-1)

grid_bagging.fit(X_train_scaled, Y_train)
best_bagging = grid_bagging.best_estimator_

print("\n Najlepsze parametry Bagging:", grid_bagging.best_params_)
print("\n Najlepsza średnia dokładność (CV):", grid_bagging.best_score_)

#8. Walidacja krzyżowa najlepszego Bagging
wyniki_cv_bagging = cross_val_score(best_bagging, X_train_scaled, Y_train, cv=5, scoring='accuracy')
print("\n Walidacja krzyżowa najlepszego Bagging:")
print(f"\n Dokładności dla każdej z 5 prób: {wyniki_cv_bagging}")
print(f"\n Średnia dokładność (Mean Accuracy): {wyniki_cv_bagging.mean():.4f}")
print(f"\n Stabilność modelu (Odchylenie std): {wyniki_cv_bagging.std():.4f}")

#9. Dopasowanie najlepszego modelu i predykcje
best_bagging.fit(X_train_scaled, Y_train)
Y_pred_train = best_bagging.predict(X_train_scaled)
Y_pred_test = best_bagging.predict(X_test_scaled)

#10. Wyniki na zbiorze uczącym
print('\n Wyniki na zbiorze uczącym:')
conf_matrix_train = confusion_matrix(Y_train, Y_pred_train)
print(conf_matrix_train)
print(classification_report(Y_train, Y_pred_train))

TN, FP, FN, TP = conf_matrix_train.ravel()
accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print(f'\n Dokładność: {accuracy}')
print(f'\n Czułość: {sensitivity}')
print(f'\n Specyficzność: {specificity}')

#11. Wyniki na zbiorze testowym
print('\n Wyniki na zbiorze testowym:')
conf_matrix_test = confusion_matrix(Y_test, Y_pred_test)
print(conf_matrix_test)
print(classification_report(Y_test, Y_pred_test))

TNt, FPt, FNt, TPt = conf_matrix_test.ravel()
accuracy_t = (TPt + TNt) / (TPt + TNt + FPt + FNt)
sensitivity_t = TPt / (TPt + FNt)
specificity_t = TNt / (TNt + FPt)
print(f'\n Dokładność: {accuracy_t}')
print(f'\n Czułość: {sensitivity_t}')
print(f'\n Specyficzność: {specificity_t}')

#12. Wizualizacja przykładowego drzewa
plot_tree(best_bagging.estimators_[0], feature_names=X_train.columns, filled=True)
plt.show()