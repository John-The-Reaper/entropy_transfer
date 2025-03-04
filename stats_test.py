import pandas as pd
import pyarrow.feather as feather
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_rel
from tqdm import tqdm
from pyinform.transferentropy import transfer_entropy
import numpy as np

from  data_manager import DataManagerCrypto

###############################################################################
# CLASSE POUR L'ANALYSE STATISTIQUE
###############################################################################

class StatisticalAnalysis:
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
        
