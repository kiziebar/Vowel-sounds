# Vovel sounds
___
Projekt na temat klasyfikacji samogłosek oraz płci na podstawie wybranych cech. 
___

##Ekstrakcja cech

Ekstrakcja cech została wykonana na podstawie próbek głosowych znajdujących się w 
folderze *src/vowels_dataset*. W celu łatwiejszego poruszania się po plikach, 
ich nazwy posiadały następującą specyfikę *płeć_samogłoska_numerpróbkidlasamogłoski.wav*.
Ekstracji cech została wykonana z użyciem implementacji własnych oraz kilku bibliotek:

- **scipy** - moduł *singal*, przetwarzanie sygnału
- **sklearn** - skalowanie
- **numpy** - operacje numeryczne, strukutry danych
- **librosa** - przetwarzanie syngału 
- **glob** - poruszanie się po folderach
- **pandas** - struktura *DataFrame*
- **os** - poruszanie się po folderach
- **antropy** - biblioteka przeznaczona do entropii

W celu wyodrębnienia cech z sygnału należy użyć metody *create_database*, znajdującej
się w pliku *src/extraction.py*. Uzyskana strutkura posiada dwa rodzaje klas tj. samogłoskę
oraz płeć, a także 9 cech tj. częstotliwość podstawowa, 4 formanty, 3 centroidy z widma
oraz entropię widmową.

---
## Klasyfikacja

Klasyfikacja została dokonana przy pomocy algorytmów pochodzących z biblioteki *sklearn*. W celu wyodrębnienia
najważniejszych cech użyto estymatorów takich jak *AdaBoostRegressor* oraz *ExtraTreesClassifier*. Natomiast klasyfikacji
dokonano przy pomocy metod KNN, Decision Tree oraz RandomForestClassifier.

---

