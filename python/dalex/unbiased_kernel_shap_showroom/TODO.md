# next steps

## edek

### ocena jakości działania
- zbadanie obciążenia estymatora: dla wybranej obserwacji `n` razy liczymy wartości SHAP i patrzymy na średnią i wariancję estymatora, porównujemy do gt jakim jest dla nas exact shap

### porównanie z innymi metodami
- z KernelSHAP, ale przy tej samej liczbie iteracji (żeby porównanie miało sens)
- to samo, ale z ustalonym limitem czasowym
- z metodą, którą dalex zaimplementował, biorącą `n` permutacji - przy określonym limicie czasowym
- (*) dla xgboosta można porównać z TreeSHAPem

### uwagi
- wszystkie eksperymenty róbmy na 2 modelach - SVM i xgboost
- spróbujmy benchmarkować na 3 datasetach: 1 z baaardzo dużą liczbą feature'ów (tam my sobie powinniśmy lepiej radzić), 1 z bardzo dużą liczbą datapointów, ale niewielką liczbą featurów (tam pewnie lepiej KernelSHAP wypadnie) + 1 sytetyczny, który wymyśli/znajdzie kuchar

## kuchar
### implementacja
- przyspieszyć działanie
- dodać opcję paired sampling
- dodać estymację wariancji estymatora 
- dodać automatyczne zatrzymanie losowania
- dodać wspieranie wielu wątków

### dataset
- wymyslić/znaleźć spoko syntetyczny dataset