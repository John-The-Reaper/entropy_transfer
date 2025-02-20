import pandas as pd
from  data_manager import DataManager
import pyarrow.feather as feather
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_rel
from tqdm import tqdm
from pyinform.transferentropy import transfer_entropy


###############################################################################
# CLASSE POUR L'ANALYSE STATISTIQUE
###############################################################################

class StatisticalAnalysis:
    @staticmethod
    def _flatten_df(df):
        """
        Aplati les colonnes de la DataFrame correspondant aux intervalles (commençant par 'H')
        en une seule liste en ignorant les valeurs manquantes.
        """
        cols = [col for col in df.columns if col.startswith('H')]
        flat = df[cols].values.flatten()
        # Filtre des valeurs None/NaN
        return [v for v in flat if pd.notnull(v)]
    
    @staticmethod
    def calculate_pearson(df1, df2):
        """
        Calcule la corrélation de Pearson entre les deux DataFrames.
        """
        flat1 = StatisticalAnalysis._flatten_df(df1)
        flat2 = StatisticalAnalysis._flatten_df(df2)
        if len(flat1) != len(flat2) or len(flat1) < 2:
            return None
        corr, _ = pearsonr(flat1, flat2)
        return corr

    @staticmethod
    def calculate_spearman(df1, df2):
        """
        Calcule la corrélation de Spearman entre les deux DataFrames.
        """
        flat1 = StatisticalAnalysis._flatten_df(df1)
        flat2 = StatisticalAnalysis._flatten_df(df2)
        if len(flat1) != len(flat2) or len(flat1) < 2:
            return None
        corr, _ = spearmanr(flat1, flat2)
        return corr

    @staticmethod
    def calculate_kendall(df1, df2):
        """
        Calcule la corrélation de Kendall entre les deux DataFrames.
        """
        flat1 = StatisticalAnalysis._flatten_df(df1)
        flat2 = StatisticalAnalysis._flatten_df(df2)
        if len(flat1) != len(flat2) or len(flat1) < 2:
            return None
        corr, _ = kendalltau(flat1, flat2)
        return corr

    @staticmethod
    def paired_t_test(df1, df2):
        """
        Effectue un test t apparié sur les données aplaties.
        Retourne la statistique t et la p-value.
        """
        flat1 = StatisticalAnalysis._flatten_df(df1)
        flat2 = StatisticalAnalysis._flatten_df(df2)
        if len(flat1) != len(flat2) or len(flat1) < 2:
            return None, None
        t_stat, p_value = ttest_rel(flat1, flat2)
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
    def transfer_entropy(serie_source, serie_cible, lags_range=100):
        """
        Calcul d'un transfer d'entropie sur deux séries de données avec un "lag" en nombre de périodes

        Closer to 1 = no entropy transfer
        """
        data = DataManager.absolute(serie_source, serie_cible)
        te_values = []


        for lag in range(lags_range):
            min_length = min(len(data[0]) - lag, len(data[1]) - lag)
            source = data[0][:min_length]
            target = data[1][lag:lag + min_length]

            te_value = transfer_entropy(source, target, k=1)  # k = ordre du passé pris en compte
            te_values.append((lag, te_value))
        return te_values


