import pandas as pd
import pyarrow.feather as feather
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_rel
from tqdm import tqdm
from pyinform.transferentropy import transfer_entropy
import numpy as np
import os
import hashlib
import json
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import time

try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba n'est pas disponible. L'installation est recommandée pour des performances optimales.")
    print("pip install numba")

from data_manager import DataManagerCrypto

###############################################################################
# CLASSE POUR L'ANALYSE STATISTIQUE
###############################################################################

class StatisticalAnalysis:
    # Dossier pour stocker les résultats de cointégration en cache
    CACHE_DIR = os.path.join("data", "cointegration_cache")
    
    @staticmethod
    def calculate_pearson(df1, df2):
        """
        Calcule la corrélation de Pearson entre les deux DataFrames.
        """
        data = DataManagerCrypto.data_modify(df1, df2)

        corr, _ = pearsonr(data[0], data[1])
        return corr

    @staticmethod
    def calculate_spearman(df1, df2):
        """
        Calcule la corrélation de Spearman entre les deux DataFrames.
        """
        data = DataManagerCrypto.data_modify(df1, df2)
        
        corr, _ = spearmanr(data[0], data[1])
        return corr

    @staticmethod
    def calculate_kendall(df1, df2):
        """
        Calcule la corrélation de Kendall entre les deux DataFrames.
        """
        data = DataManagerCrypto.data_modify(df1, df2)

        corr, _ = kendalltau(data[0], data[1])
        return corr

    @staticmethod
    def paired_t_test(df1, df2):
        """
        Effectue un test t apparié sur les données aplaties.
        Retourne la statistique t et la p-value.
        """
        data = DataManagerCrypto.data_modify(df1, df2)
        
        t_stat, p_value = ttest_rel(data[0], data[1])
        return t_stat, p_value

    @staticmethod
    def calculate_correlation_gap(df_ref, df_sec, sample_days=14, time_gap=20):
        """
        Calcule la corrélation de Pearson entre deux ensembles de données en décalant
        df_sec de 'time_gap' jours par rapport à df_ref.
        Chaque DataFrame doit contenir une ligne par jour.
        """
        if len(df_ref) < sample_days or len(df_sec) < sample_days + time_gap:
            print("Données insuffisantes pour l'analyse demandée.")
            return None

        # On considère que chaque ligne représente un jour, on découpe donc la DataFrame
        sample_ref = df_ref.iloc[:sample_days]
        sample_sec = df_sec.iloc[time_gap:sample_days+time_gap]
        return StatisticalAnalysis.calculate_pearson(sample_ref, sample_sec)

    @staticmethod
    def transfer_entropy(series_source, series_target, lags_range=100):
        """
        Calcul optimisé du transfert d'entropie pour une gamme de lags.
        """
        data = DataManagerCrypto.data_modify(series_source, series_target, entropy=True)

        source = np.array(data[0])
        target = np.array(data[1])

        # Vérification des tailles minimales requises
        min_length = min(len(source), len(target))
        if min_length < lags_range:
            print("La longueur des séries est inférieure au lags_range.")
            lags_range = min_length

        te_values = np.zeros(lags_range)

        for lag in range(lags_range):
            truncated_source = source[: min_length - lag]
            truncated_target = target[lag : min_length]

            try:
                te_values[lag] = transfer_entropy(truncated_source, truncated_target, k=1)
            except Exception as e:
                print(f"Erreur de calcul TE pour le lag {lag}: {e}")
                te_values[lag] = np.nan  # Marquer les erreurs

        return list(enumerate(te_values))
    
    @staticmethod
    def compute_te_over_time(series_source, series_target, window_size=100):
        """
        Calcule l'évolution du transfert d'entropie sur une fenêtre glissante.
        """
        data = DataManagerCrypto.data_modify(series_source, series_target, entropy=True)
        source = np.array(data[0])
        target = np.array(data[1])

        min_length = min(len(source), len(target))
        if min_length < window_size:
            raise ValueError("La taille des séries est inférieure à la fenêtre d'analyse.")

        te_values = [
            transfer_entropy(source[i:i+window_size], target[i:i+window_size], k=1)
            for i in range(min_length - window_size)
        ]

        return te_values

    
    @staticmethod
    def moving_average(values, window_size):
        """Calcule une moyenne mobile qui s'étend jusqu'à la dernière valeur."""
        moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        
        # Remplissage des premières valeurs pour garder la même taille
        start_vals = [np.mean(values[:i+1]) for i in range(window_size - 1)]
        full_moving_avg = np.concatenate([start_vals, moving_avg])
    
        return full_moving_avg
    
    @staticmethod
    def _get_cache_key(X, Y, max_lags):
        """Génère une clé de cache unique pour un test de cointégration"""
        # Prendre un échantillon pour réduire le temps de calcul du hash
        sample_size = min(1000, len(X))
        if sample_size < len(X):
            X_sample = X[:sample_size//2].tolist() + X[-sample_size//2:].tolist()
            Y_sample = Y[:sample_size//2].tolist() + Y[-sample_size//2:].tolist()
        else:
            X_sample = X.tolist()
            Y_sample = Y.tolist()
            
        data_str = str(X_sample) + str(Y_sample) + str(max_lags)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    @staticmethod
    def _check_cointegration_cache(cache_key):
        """Vérifie si un résultat de cointégration est en cache"""
        os.makedirs(StatisticalAnalysis.CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(StatisticalAnalysis.CACHE_DIR, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    @staticmethod
    def _save_to_cointegration_cache(cache_key, result):
        """Sauvegarde un résultat de cointégration en cache"""
        os.makedirs(StatisticalAnalysis.CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(StatisticalAnalysis.CACHE_DIR, f"{cache_key}.json")
        
        try:
            # Créer une copie du résultat pour éviter de modifier l'original
            result_copy = dict(result)
            
            # Convertir les ndarray en liste pour la sérialisation JSON
            if 'critical_values' in result_copy and hasattr(result_copy['critical_values'], 'tolist'):
                result_copy['critical_values'] = result_copy['critical_values'].tolist()
            
            # Convertir explicitement les booléens numpy en booléens Python
            if 'cointegrated' in result_copy and not isinstance(result_copy['cointegrated'], bool):
                result_copy['cointegrated'] = bool(result_copy['cointegrated'])
            
            # Convertir toutes les valeurs numériques en type Python standard
            for key, value in result_copy.items():
                if hasattr(value, 'item'):  # Détecte les types numpy comme np.float64, np.int64, etc.
                    result_copy[key] = value.item()  # Convertit en type Python standard
            
            with open(cache_file, 'w') as f:
                json.dump(result_copy, f)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du cache: {e}")
            # Continuer sans cacher le résultat
    
    @staticmethod
    def _fast_coint(X, Y, maxlag=100):
        """Version optimisée du test de cointégration"""
        # Test de cointégration d'Engle-Granger avec paramètres optimisés
        # Utiliser autolag=None est beaucoup plus rapide mais moins précis
        test_result = coint(X, Y, maxlag=maxlag, trend='c', autolag=None)
        test_stat, p_value, critical_values = test_result
        
        # Calcul rapide de la relation de long terme en utilisant numpy
        # C'est beaucoup plus rapide que statsmodels OLS
        X_with_const = np.column_stack((np.ones(len(X)), X))
        
        # Utiliser l'algèbre linéaire de numpy pour une résolution plus rapide que OLS
        beta = np.linalg.lstsq(X_with_const, Y, rcond=None)[0][1]
        
        # L'hypothèse nulle est l'absence de cointégration
        # Donc une p-value faible (< 0.05) indique la cointégration
        is_cointegrated = p_value < 0.05
        
        return {
            'cointegrated': is_cointegrated,
            'p_value': p_value,
            'test_statistic': test_stat,
            'critical_values': critical_values,
            'beta': beta
        }
        
    @staticmethod
    def test_cointegration(df1, df2, max_lags=100, use_cache=True):
        """
        Teste la cointégration entre deux séries temporelles en utilisant le test de Engle-Granger.
        Version optimisée avec mise en cache des résultats.
        
        Args:
            df1, df2: DataFrames contenant les séries temporelles (prix)
            max_lags: Nombre maximal de lags pour le test de stationnarité
            use_cache: Si True, utilise le cache pour les résultats déjà calculés
            
        Returns:
            Un dictionnaire contenant:
            - 'cointegrated': bool indiquant si les séries sont cointégrées
            - 'p_value': p-value du test
            - 'test_statistic': statistique du test
            - 'beta': coefficient de la relation de long terme
        """
        # Préparation des données - utiliser directement les séries pandas
        # S'assurer que les données sont des arrays numpy
        X = np.array(df1.values)
        Y = np.array(df2.values)
        
        # Vérifier que les données ne contiennent pas de NaN
        if np.isnan(X).any() or np.isnan(Y).any():
            raise ValueError("Les données contiennent des valeurs NaN")
            
        if use_cache:
            # Vérifier si le résultat est en cache
            cache_key = StatisticalAnalysis._get_cache_key(X, Y, max_lags)
            cached_result = StatisticalAnalysis._check_cointegration_cache(cache_key)
            
            if cached_result:
                return cached_result
        
        # Si pas en cache ou cache désactivé, calculer le résultat
        result = StatisticalAnalysis._fast_coint(X, Y, maxlag=max_lags)
        
        # Mettre en cache le résultat si nécessaire
        if use_cache:
            StatisticalAnalysis._save_to_cointegration_cache(cache_key, result)
            
        return result
    
    @staticmethod
    def _process_pair(pair_data, max_lags=100, use_cache=True):
        """Traite une paire pour le test de cointégration (utilisé pour la parallélisation)"""
        symbol1, symbol2, df1, df2 = pair_data
        try:
            # S'assurer que les deux séries ont la même longueur
            common_index = df1.index.intersection(df2.index)
            if len(common_index) < 30:  # Au moins 30 points de données communs
                return None
                
            df1_common = df1.loc[common_index]
            df2_common = df2.loc[common_index]
            
            # Effectuer le test de cointégration
            coint_result = StatisticalAnalysis.test_cointegration(df1_common, df2_common, max_lags, use_cache)
            
            # Créer le résultat
            result = {
                'pair': f"{symbol1}_{symbol2}",
                'symbol1': symbol1,
                'symbol2': symbol2,
                'cointegrated': coint_result['cointegrated'],
                'p_value': coint_result['p_value'],
                'test_statistic': coint_result['test_statistic'],
                'beta': coint_result['beta']
            }
            
            return result
        except Exception as e:
            print(f"Erreur lors du test de cointégration pour {symbol1} et {symbol2}: {e}")
            return None
    
    @staticmethod
    def perform_cointegration_analysis(symbols, timeframe='1d', results_folder='cointegration_results', 
                                      start_date=None, end_date=None, max_workers=None, use_cache=True,
                                      max_retries=3):
        """
        Effectue une analyse de cointégration entre toutes les paires possibles de symboles.
        Version optimisée avec parallélisation et mise en cache.
        
        Args:
            symbols: Liste de symboles à tester (par exemple ["BTC/USDT", "ETH/USDT", ...])
            timeframe: Intervalle de temps pour les données (ex: 1d, 4h, 1h, 15m)
            results_folder: Dossier où sauvegarder les résultats
            start_date: Date de début en timestamp milliseconds
            end_date: Date de fin en timestamp milliseconds
            max_workers: Nombre maximum de processus parallèles (None = nombre de CPU)
            use_cache: Si True, utilise le cache pour les résultats déjà calculés
            max_retries: Nombre maximum de tentatives en cas d'échec
            
        Returns:
            DataFrame contenant les résultats de tous les tests
        """
        import os
        from itertools import combinations
        from datetime import datetime, timedelta
        
        start_time = time.time()
        
        # Création du dossier de résultats s'il n'existe pas
        os.makedirs(results_folder, exist_ok=True)
        
        # Initialisation du gestionnaire de données
        manager = DataManagerCrypto()
        
        # Définir les dates de début et de fin par défaut si non spécifiées
        if start_date is None:
            start_date = int(datetime(2022, 1, 1).timestamp() * 1000)  # 1er janvier 2022
            
        if end_date is None:
            end_date = int(datetime.now().timestamp() * 1000)  # Aujourd'hui
            
        # Définir le nombre de processus si non spécifié
        if max_workers is None:
            max_workers = cpu_count()
            print(f"Optimisation: utilisation de {max_workers} processus parallèles")
        else:
            print(f"Optimisation: utilisation de {max_workers} processus parallèles (défini par l'utilisateur)")
            
        # Téléchargement des données pour tous les symboles
        print(f"Téléchargement des données pour {len(symbols)} paires en {timeframe}...")
        
        # Dictionnaire pour stocker les DataFrames
        dataframes = {}
        
        # Fonction pour télécharger une paire avec gestion des erreurs et tentatives
        def download_with_retry(symbol, retry_count=0):
            try:
                candles = manager.fetch_historical_data(symbol, timeframe, start_date, end_date, use_cache=True)
                
                # Vérifier si des données ont été récupérées
                if not candles or len(candles) < 30:  # Au moins 30 points de données pour une analyse significative
                    if retry_count < max_retries and not candles:
                        print(f"Échec de téléchargement pour {symbol}, tentative {retry_count+1}/{max_retries}...")
                        return download_with_retry(symbol, retry_count + 1)
                    else:
                        print(f"Données insuffisantes pour {symbol}, ignoré.")
                        return None
                
                # Conversion en DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df['close']  # On ne garde que les prix de clôture
            except Exception as e:
                if retry_count < max_retries:
                    print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
                    print(f"Nouvelle tentative {retry_count+1}/{max_retries}...")
                    return download_with_retry(symbol, retry_count + 1)
                else:
                    print(f"Échec après {max_retries} tentatives pour {symbol}: {e}")
                    return None
        
        # Utilisation de ThreadPoolExecutor pour télécharger les données en parallèle
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Préparer les futures
            future_to_symbol = {executor.submit(download_with_retry, symbol): symbol for symbol in symbols}
                
            # Traiter les résultats au fur et à mesure
            for future in tqdm(as_completed(future_to_symbol), total=len(symbols), 
                               desc="Téléchargement des données", unit="paire", colour="blue"):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None:
                        dataframes[symbol] = df
                except Exception as e:
                    print(f"Erreur non gérée pour {symbol}: {e}")
        
        # Vérifier si nous avons suffisamment de paires
        if len(dataframes) < 2:
            print(f"Pas assez de données téléchargées. Seulement {len(dataframes)} symboles disponibles sur {len(symbols)}.")
            return pd.DataFrame()
        
        # Test de cointégration pour toutes les paires possibles
        print(f"\nPréparation des paires pour les tests de cointégration...")
        pairs_data = []
        
        # Générer toutes les combinaisons possibles de paires
        pairs = list(combinations(list(dataframes.keys()), 2))
        total_pairs = len(pairs)
        
        if total_pairs == 0:
            print("Aucune paire à tester. Vérifiez vos données.")
            return pd.DataFrame()
        
        print(f"Préparation de {total_pairs} paires pour l'analyse de cointégration...")
        
        for symbol1, symbol2 in pairs:
            pairs_data.append((symbol1, symbol2, dataframes[symbol1], dataframes[symbol2]))
        
        # Effectuer les tests de cointégration en parallèle
        print(f"Lancement des tests de cointégration sur {total_pairs} paires...")
        
        results = []
        
        # Utiliser multiprocessing.Pool pour la parallélisation
        with Pool(processes=max_workers) as pool:
            # Fonction partielle avec les paramètres fixes
            process_func = partial(StatisticalAnalysis._process_pair, use_cache=use_cache)
            
            # Utiliser imap_unordered pour traiter les paires en parallèle avec une barre de progression
            for result in tqdm(pool.imap_unordered(process_func, pairs_data), 
                               total=len(pairs_data), desc="Tests de cointégration", 
                               unit="paire", colour="green"):
                if result is not None:
                    results.append(result)
        
        # Création du DataFrame de résultats
        if not results:
            print("Aucun résultat de cointégration n'a été obtenu.")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        
        # Tri par p-value pour trouver les paires les plus cointégrées
        results_df = results_df.sort_values(by='p_value')
        
        # Sauvegarde des résultats
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_folder, f'cointegration_results_{timeframe}_{timestamp}.csv')
        results_df.to_csv(results_file, index=False)
        
        # Calcul du temps d'exécution
        execution_time = time.time() - start_time
        minutes, seconds = divmod(execution_time, 60)
        
        print(f"\nAnalyse terminée en {int(minutes)} minutes et {int(seconds)} secondes")
        print(f"\nRésultats sauvegardés dans {results_file}")
        print(f"\nPaires cointégrées trouvées: {results_df['cointegrated'].sum()} sur {len(results_df)}")
        
        return results_df
        