import shap
import dalex as dx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

#Celem projektu jest analiza danych medycznych i przewidywanie ryzyka choroby
#oczu na podstawie wybranych cech pacjentów, z wykorzystaniem różnych
#metod uczenia maszynowego oraz interpretowalności modeli

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

#3. Sprawdzenie występowania braków danych

print("\n Wartości brakujące:")
print(dane.isnull().sum()) #Nie występują braki danych, więc nie jest konieczne uzupełnianie wartości

#4. Wstępna analiza wykorzystywanych danych

print("\n Typy danych poddanych analizie:")
print(dane.dtypes)

#5. Podstawowe statystyki

print("\n Podstawowy opis danych:")
print(dane.describe())

#6. Poziom zbalansowania zbioru danych

print("\n Procentowy udział osób z chorobą oczu (True) i bez (False):")
print(dane["has_eye_disease"].value_counts(normalize = True))

print("\n Ilościowy udział osób z chorobą oczu (True) i bez (False):")
print(dane["has_eye_disease"].value_counts())

#7. Modyfikacja zmiennych kategorycznych na numeryczne

#Funkcja get dummies - automatycznie dzieli kolumny z więcej niz 2 zmiennymi, rozdziela je na osobne kolumny i przyporządkowuje im wartości tak albo nie
dane = (pd.get_dummies (dane, columns=['bp_category','obesity_group', ], drop_first=True))

#Zmiana zmiennych tak i nie na zmienne liczbowe 1 i 0
dane_najwyst = dane.astype({col: 'int' for col in dane.select_dtypes('bool').columns})
print('\n Prezentacja typów danych poddanych analizie po zastosowaniu modyfikacji zmiennych na numeryczne:')
print(dane_najwyst.dtypes)

#W tym momęcie mamy przygotowane dane do dalszych obliczeń - usunięte zbędne kolumny, dane zmienione na zmienne numeryczne
print('\n Dane po modyfikacji - zmienne numeryczne:')
print(dane.head())

#8. Korelacja zmiennych oraz analiza wpływu poszczególnych zmiennych na zmienną prognozowaną Y

#Zmienne numeryczne bez zmiennej docelowej 'has_eye_disease'
num_cols = dane_najwyst.select_dtypes(include=['int64', 'float64']).columns.drop('has_eye_disease')

#8.1. Heatmapa korelacji zmiennych
plt.figure(figsize=(12,10))  # ustalamy rozmiar wykresu
correlation_matrix = dane_najwyst.corr()  # macierz korelacji wszystkich zmiennych
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")  # rysujemy heatmapę
plt.title("Heatmapa korelacji zmiennych")
plt.show()

#8.2. Analiza wpływu Top 10 zmiennych na zmienną prognozowaną Y
num_colsW = [col for col in dane_najwyst.select_dtypes(include=['int64','float64']).columns if col != 'has_eye_disease']
correlations = dane_najwyst[num_colsW + ['has_eye_disease']].corr()['has_eye_disease'].drop('has_eye_disease')
top10_cols = correlations.abs().sort_values(ascending=False).head(10)

#Przedstawienie wyników w postaci wykresu słupkowego
plt.figure(figsize=(8,5))
top10_cols.plot(kind='bar', color='skyblue')
plt.ylabel('Korelacja z has_eye_disease')
plt.title('Top 10 zmiennych wpływających na chorobę oczu - zmienną prognozowaną has_eye_disease')
plt.show()

#9. Wartości odstające
#Ze względu na charakter danych poddanych analizie - dane medyczne - zdecydowałyśmy na nieususwanie wartości odstających
#Wynika to z faktu, że stanowią one informację o stanie zdrowia, a nie błąd obliczeniowy

Q1 = dane_najwyst[num_cols].quantile(0.25) #https://mateuszgrzyb.pl/3-metody-wykrywania-obserwacji-odstajacych-w-python/
Q3 = dane_najwyst[num_cols].quantile(0.75)
IQR = Q3 - Q1

dolna = Q1 - 1.5 * IQR
gorna = Q3 + 1.5 * IQR

wystajace = (dane_najwyst[num_cols] < dolna) + (dane_najwyst[num_cols] > gorna)

print("\n Liczba wartości odstających w kolumnach:")
print(wystajace.sum())

#10. Podział na X i Y - z pominięciem 'has_eye_disease'

X = dane_najwyst.drop('has_eye_disease', axis = 1)
Y = dane_najwyst['has_eye_disease']

#Musimy to zrobić tutaj, bo StandardScaler zamieni dane na tablicę bez nazw
kolumny_nazwy = X.columns

#11. Podział na zbiór uczący i testowy

X_train, X_test, Y_train, Y_test = (train_test_split (X, Y, test_size=0.2, random_state=50))

#12. Skalowanie cech - każda cecha ma średnią 0 i odchylenie standardowe 1 - bez tego model mógłby uznać, że zmienna o większych wartościach jest ważniejsza

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=kolumny_nazwy)
X_test = pd.DataFrame(scaler.transform(X_test), columns=kolumny_nazwy)

#13. Optymalizacja hiperparametrów - regresja logistyczna

param_grid = {'C': [0.1, 1, 10]}

grid = GridSearchCV(LogisticRegression(max_iter=10000, class_weight='balanced'), param_grid,cv=5,scoring='accuracy')

grid.fit(X_train, Y_train)

print("\n Najlepsze parametry dla regresji logistycznej:", grid.best_params_)

RL = grid.best_estimator_

#14. Walidacja krzyżowa końcowego modelu, wykonujemy ją na zbiorze treningowym (podział na 5 części)

wyniki_cv = cross_val_score(RL, X_train, Y_train, cv=5, scoring='accuracy')

print("\n Walidacja krzyżowa (model po GridSearch):")
print(f"\n Dokładności: {wyniki_cv}")
print(f"\n Średnia dokładność: {wyniki_cv.mean()}")
print(f"\n Odchylenie std: {wyniki_cv.std()}")

RL.fit(pd.DataFrame(X_train, columns=X.columns), Y_train)

#Predykcje dla zbiorów
y_pred_train = RL.predict(X_train)
y_pred_test = RL.predict(X_test)

print("\n Macierz błędu zbiór uczący:")
macierz_błędu_train = confusion_matrix(Y_train, y_pred_train)
print(macierz_błędu_train)

TN_tr, FP_tr, FN_tr, TP_tr = macierz_błędu_train.ravel()
dokladnosc_tr = (TP_tr + TN_tr) / (TP_tr + TN_tr + FP_tr + FN_tr)
czulosc_tr = TP_tr / (TP_tr + FN_tr)
specyficznosc_tr = TN_tr / (TN_tr + FP_tr)

print(f"\n Dokładność (Train): {dokladnosc_tr}")
print(f"\n Czułość (Train): {czulosc_tr}")
print(f"\n Specyficzność (Train): {specyficznosc_tr}")

print("\n Raport klasyfikacji dla zbioru uczącego:")
print(classification_report(Y_train, y_pred_train))

print("\n Macierz błędu zbiór testowy")
macierz_błędu = confusion_matrix(Y_test, y_pred_test)
print(macierz_błędu)

TN, FP, FN, TP = macierz_błędu.ravel()
dokładność = (TP + TN) / (TP + TN + FP + FN)
czułość = TP / (TP + FN)
specyficzność = TN / (TN + FP)

print(f"\n Dokładność: {dokładność}")
print(f"\n Czułość:{czułość}")
print(f"\n Specyficzność:{specyficzność}")

print("\n Raport klasyfikacji dla zbioru testowego:")
print(classification_report(Y_test, y_pred_test))

#Wykonujemy to po treningu, żeby zobaczyć co model uznał za najważniejsze
parametry = pd.DataFrame({
    'zmienna': kolumny_nazwy,
    'kierunek': RL.coef_[0], #Współczynniki regresji logistycznej dla każdej zmienne - jak zmiana danej zmiennej wpływa na szanse choroby oczu
    'siła': np.exp(RL.coef_[0]) #O ile zmienia się ryzyko choroby przy jednostkowej zmianie zmiennej
})

#kierunek - dodatni (wzrost ryzyka), ujemny - (spadek ryzyka)
print('\n Ważność zmiennych według regresji logistycznej:')
print(parametry)

#15. Analiza interpretowalności - SHAP - dla regresji logistycznej
#wykres pokazuje wszystkie cechy naraz, kolor oznacza wartość zmiennej, a pozycja w osi X wpływ na predykcję

X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

RL.fit(X_train_df, Y_train)

explainer = shap.LinearExplainer(RL, X_train_df)
shap_values = explainer.shap_values(X_test_df)
shap.summary_plot(shap_values, X_test_df)

#16. Analiza interpretowalności - Wykres częściowej zależności - dla has_diabetic_retinopathy
#Sprawdzamy zależność predykcji od konkretnej cechy
feature_idx = X.columns.get_loc('has_diabetic_retinopathy')
PartialDependenceDisplay.from_estimator(RL, X_test, [feature_idx], feature_names=X.columns)

plt.show()

#17. Analiza interpretowalności - profile ceteris paribus
#Tworzymy dla wytrenowanego modelu RL - regresji logistycznej

explainer_cp = dx.Explainer(model=RL, data=X_test_df, y=Y_test, label="Regresja logistyczna")

#Wybieramy jednego pacnejta do analixt
observation = X_test_df.iloc[0]

#Tworzymy profil ceteris paribus
cp_profile = explainer_cp.predict_profile(observation)

#Wykres dla zmiennej 'age' - jej wpływ na predykcje dla wybranej obserwacji (pacjenta)
cp_profile.plot(variables=["age"])
plt.show()
