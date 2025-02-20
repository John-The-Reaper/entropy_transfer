import ccxt
import os
from datetime import datetime, timedelta
from time import sleep
import pandas as pd
import pyarrow.feather as feather
from tqdm import tqdm

from data_manager import DataManager
from stats_test import StatisticalAnalysis


###############################################################################
# CONFIGURATION GLOBALE
###############################################################################
TIMEFRAME = '1h'               # Intervalle pour les chandeliers



def main():
    # Exemple 1 : Récupération et sauvegarde des données pour deux symboles
    dm = DataManager()
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'ATOM/USDT', 'INJ/USDT', 'DYM/USDT']  # format ccxt "BASE/QUOTE"
    weeks = 12  # Nombre de semaines de données à récupérer

    for symbol in symbols:
        df = dm.get_data(symbol, weeks=weeks, timeframe=TIMEFRAME)
        # On retire le caractère "/" pour le nom du fichier
        symbol_filename = symbol.replace('/', '')
        dm.save_to_feather(df, symbol_filename)

    """
    # Exemple 2 : Lecture des données et analyse statistique
    ref_symbol = "BTCUSDT"
    sec_symbol = "ETHUSDT"
    df_ref = DataManager.read_from_feather(ref_symbol)
    df_sec = DataManager.read_from_feather(sec_symbol)
    
    # Calcul de différentes corrélations sur l'ensemble des jours
    pearson_corr = StatisticalAnalysis.calculate_pearson(df_ref, df_sec)
    spearman_corr = StatisticalAnalysis.calculate_spearman(df_ref, df_sec)
    kendall_corr = StatisticalAnalysis.calculate_kendall(df_ref, df_sec)
    t_stat, p_value = StatisticalAnalysis.paired_t_test(df_ref, df_sec)

    print(f"Corrélation de Pearson entre BTC et ETH : {pearson_corr}")
    print(f"Corrélation de Spearman entre BTC et ETH : {spearman_corr}")
    print(f"Corrélation de Kendall entre BTC et ETH : {kendall_corr}")
    print(f"Test t apparié : t = {t_stat}, p-value = {p_value}")

    # Calcul d'une corrélation sur un échantillon décalé
    sample_days = 14
    time_gap = 20
    corr_gap = StatisticalAnalysis.calculate_correlation_gap(df_ref, df_sec,
                                                               sample_days=sample_days,
                                                               time_gap=time_gap)
    print(f"Corrélation sur {sample_days} jours avec un décalage de {time_gap} jours : {corr_gap}")
    """

if __name__ == '__main__':
    main()
