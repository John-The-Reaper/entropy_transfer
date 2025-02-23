# Crypto Correlation & Transfer Entropy Analysis

## Description
Ce projet permet d'analyser la corrélation et le transfert d'entropie entre différents actifs financiers en utilisant des données de l'API Binance. Il inclut plusieurs outils pour :

- Récupérer et stocker les données de marché sous format `feather`.
- Calculer la corrélation (Pearson, Spearman, Kendall) entre les actifs.
- Analyser le transfert d'entropie pour identifier les influences directionnelles entre actifs.
- Visualiser les résultats sous forme de graphiques.

## Fonctionnalités
✅ **Récupération des données** via `ccxt` et stockage optimisé avec `feather`.
✅ **Calculs statistiques avancés** (corrélation, tests t, transfert d'entropie).
✅ **Optimisation des performances** grâce à `numpy`.
✅ **Ajout de logs** pour un suivi efficace des erreurs et de l'exécution.

## Installation
### 1️⃣ Cloner le projet
```bash
git clone https://github.com/votre-repo/crypto-correlation.git
cd crypto-correlation
```
### 2️⃣ Installer les dépendances
Utilisez `pip` pour installer les bibliothèques nécessaires :
```bash
pip install -r requirements.txt
```

## Utilisation
### Récupération des données historiques
```python
from data_manager import DataManager

manager = DataManager()
data = manager.fetch_historical_data("BTC/USDT", "1d", start_ts, end_ts)
```

### Calcul de la corrélation
```python
from statistical_analysis import StatisticalAnalysis

correlation = StatisticalAnalysis.calculate_correlation(series1, series2)
```

### Analyse du transfert d'entropie
```python
te_results = StatisticalAnalysis.transfer_entropy(series1, series2, lags_range=100)
```

### Visualisation des résultats
```python
from visualization import Visualization
Visualization.plot_correlation(correlation)
```

## Configuration des logs
Le projet utilise `logging` pour tracer les erreurs et avertissements. Les logs sont enregistrés dans `app.log`.

## Contributeurs
👤 **Votre Nom** - Développeur principal

## Licence
Ce projet est sous licence MIT.

