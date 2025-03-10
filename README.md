## Description

Ce projet fournit une boîte à outils puissante pour l'analyse statistique des marchés financiers, en particulier des cryptomonnaies. Il permet d'étudier les relations entre différents actifs à travers une variété de méthodes statistiques avancées.

Le framework est conçu pour être :
- **Évolutif** : Facilement extensible avec de nouveaux tests statistiques
- **Performant** : Optimisé avec mise en cache et parallelisation
- **Intuitif** : API simple pour l'analyse de données financières

## Fonctionnalités Principales

### Récupération et Gestion des Données
- ✅ Collecte de données via l'API Binance avec `ccxt`
- ✅ Stockage optimisé au format `feather`
- ✅ Prétraitement automatique des séries temporelles

### Tests Statistiques
- ✅ **Corrélation** : Pearson, Spearman, Kendall
- ✅ **Tests de Cointégration** : Engle-Granger, Johansen
- ✅ **Transfert d'Entropie** : Mesure des influences directionnelles
- ✅ **Tests de Stationnarité** : ADF, KPSS
- ✅ **Tests de Causalité** : Granger
- 🔜 _Nouveaux tests prévus à venir_

### Visualisation
- ✅ Matrices de corrélation
- ✅ Heatmaps de cointégration
- ✅ Graphiques de paires cointégrées
- ✅ Visualisation du transfert d'entropie

## Structure du Projet

```
entropy_transfer/
│
├── data_manager.py     # Gestion des données
├── stats_test.py       # Tests statistiques et analyses
├── entropie.py         # Calculs de transfert d'entropie
├── requirements.txt    # Dépendances
└── test.py             # Tests et démonstrations
```

## Installation

### 1️⃣ Cloner le projet
```bash
git clone https://github.com/John-The-Reaper/entropy_transfer.git
cd entropy_transfer
```

### 2️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation

### Récupération des données
```python
from data_manager import DataManagerCrypto

manager = DataManagerCrypto()
data = manager.fetch_historical_data("BTC/USDT", "1h", start_ts, end_ts)
```

### Analyse statistique
```python
from stats_test import StatisticalAnalysis

# Initialiser avec mise en cache
stats = StatisticalAnalysis(cache_dir="./cache")

# Tests de corrélation
correlation = stats.calculate_correlation(series1, series2)

# Test de cointégration
coint_result = stats.test_cointegration(series1, series2)

# Transfert d'entropie
te_results = stats.transfer_entropy(series1, series2, lags_range=100)
```

### Visualisation des résultats
```python
# Créer une heatmap de cointégration
create_cointegration_heatmap(results_df, symbols, "results_folder", "1h")

# Tracer des paires cointégrées
plot_cointegrated_pair("BTC/USDT", "ETH/USDT", "results_folder", "1h", "2023-01-01", "2023-06-01")
```

## Développements Futurs

Le module de tests statistiques (`stats_test.py`) est conçu pour être étendu avec de nouveaux tests et méthodes :

- 🔜 Tests de rupture structurelle
- 🔜 Modèles ARCH/GARCH pour la volatilité
- 🔜 Tests de cointégration non-linéaire
- 🔜 Méthodes de machine learning pour la détection de relations
- 🔜 Optimisation pour les grands ensembles de données

## Contributeurs

👤 **John-The-Reaper** - Développeur principal  
👤 **Jimmy7892** - Optimisation des performances et nouvelles fonctionnalités

## Licence

Ce projet est disponible sous licence MIT.
