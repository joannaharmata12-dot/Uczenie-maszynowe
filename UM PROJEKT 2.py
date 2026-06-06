import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

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
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

#7. Optymalizacja hiperparametrów dla SVM
param_grid_svm = { 'C': [0.1, 1, 10], 'kernel': ['linear'], 'class_weight': ['balanced']}

grid_svm = GridSearchCV( estimator=svm.SVC(),
                         param_grid=param_grid_svm,
                         scoring='roc_auc', #model na podstawie pola pod krzywą ROC
                         cv=5, #dane dzielone są na 5 części
                         n_jobs=-1) #używa wszystkich dostępnych cores

#Trenowanie GridSearch
grid_svm.fit(X_train_scaled, Y_train)
best_svm = grid_svm.best_estimator_

print("\n Najlepsze parametry SVM:", grid_svm.best_params_)
print("\n Najlepsza średnia AUC (CV):", grid_svm.best_score_)

#8. Walidacja krzyżowa dla najlepszego SVM
wyniki_cv_svm = cross_val_score(best_svm, X_train_scaled, Y_train, cv=5, scoring='accuracy')
print("\n Walidacja krzyżowa dla najlepszego SVM:")
print(f"\n Dokładności dla każdej z 5 prób: {wyniki_cv_svm}")
print(f"\n Średnia dokładność (Mean Accuracy): {wyniki_cv_svm.mean():.4f}")
print(f"\n Stabilność modelu (Odchylenie std): {wyniki_cv_svm.std():.4f}")

#9. Predykcje
y_pred_train = best_svm.predict(X_train_scaled)
y_pred_test = best_svm.predict(X_test_scaled)

#10. Wyniki na zbiorze uczącym
print("\n Macierz pomyłek (zbiór uczący):")
conf_matrix_train = confusion_matrix(Y_train, y_pred_train)
print(conf_matrix_train)
print("\n Raport klasyfikacji (zbiór uczący):")
print(classification_report(Y_train, y_pred_train))

TN, FP, FN, TP = conf_matrix_train.ravel()
dokladnosc = (TP + TN) / (TP + TN + FP + FN)
czulosc = TP / (TP + FN)
specyficznosc = TN / (TN + FP)
print(f"\n Dokładność: {dokladnosc}")
print(f"\n Czułość: {czulosc}")
print(f"\n Specyficzność: {specyficznosc}")

#11. Wyniki na zbiorze testowym
print("\n Macierz pomyłek (zbiór testowy):")
conf_matrix_test = confusion_matrix(Y_test, y_pred_test)
print(conf_matrix_test)
print("\n Raport klasyfikacji (zbiór testowy):")
print(classification_report(Y_test, y_pred_test))

TNt, FPt, FNt, TPt = conf_matrix_test.ravel()
dokladnosc_t = (TPt + TNt) / (TPt + TNt + FPt + FNt)
czulosc_t = TPt / (TPt + FNt)
specyficznosc_t = TNt / (TNt + FPt)
print(f"\n Dokładność: {dokladnosc_t}")
print(f"\n Czułość: {czulosc_t}")
print(f"\n Specyficzność: {specyficznosc_t}")
