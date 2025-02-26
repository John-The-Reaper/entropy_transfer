import ccxt
import os
from datetime import datetime, timedelta
from time import sleep
import numpy as np
import pandas as pd
import math
import pyarrow.feather as feather
from tqdm import tqdm


###############################################################################
# CLASSE DE GESTION DES DONNÉES
###############################################################################

INTERVALS_PER_DAY = 24  # 24 intervalles par jour (timeframe de 1h)
DATA_DIR = "data"        # Répertoire de sauvegarde des Feather

class DataManagerCrypto:
    def __init__(self, exchange_name='binance'):
        # Crée une instance ccxt avec gestion automatique du rate limit
        self.exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    
    def fetch_historical_data(self, symbol, timeframe, start_ts, end_ts):
        """
        Récupère les chandeliers depuis start_ts jusqu'à end_ts (en millisecondes)
        en paginant si nécessaire.
        """
        all_candles = []
        since = start_ts
        limit = 1000  # limite de chandeliers par appel
        pbar = tqdm(desc=f"Fetching {symbol}", unit="candle")

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            except Exception as e:
                print(f"Erreur lors de la récupération de {symbol}: {e}")
                break

            if not candles:
                break

            for candle in candles:
                ts = candle[0]
                if ts > end_ts:
                    # On arrête si l'on dépasse la date de fin
                    break
                all_candles.append(candle)

            pbar.update(len(candles))
            last_ts = candles[-1][0]
            if last_ts >= end_ts:
                break
            since = last_ts + 1
            sleep(self.exchange.rateLimit / 1000)
        pbar.close()
        return all_candles

    def get_data(self, symbol, timeframe, weeks=4 ):
        """
        Récupère les données sur une période (en semaines) et les organise dans
        une DataFrame avec les colonnes :
          - Week, Day, puis les valeurs de chaque intervalle en fonction de la timeframe
        Le changement est calculé par : (close - open) / open * 100.
        """
        now_dt = datetime.now()
        start_dt = now_dt - timedelta(weeks=weeks)
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(now_dt.timestamp() * 1000)

        print(f"Récupération de {symbol} depuis {start_dt} jusqu'à {now_dt}...")
        candles = self.fetch_historical_data(symbol, timeframe, start_ts, end_ts)
        
        # Initialisation de la structure : liste de listes (weeks x 7 jours)
        data_structure = [
            [[None for _ in range(INTERVALS_PER_DAY)] for _ in range(7)]
            for _ in range(weeks)
        ]

        for candle in candles:
            ts, open_price, high, low, close_price, volume = candle
            candle_dt = datetime.fromtimestamp(ts / 1000)
            if open_price:
                pct_change = ((close_price - open_price) / open_price) * 100
            else:
                pct_change = None

            # Calcul du décalage en jours par rapport à start_dt
            delta_days = (candle_dt.date() - start_dt.date()).days
            if delta_days < 0 or delta_days >= weeks * 7:
                continue

            # On classe la donnée dans la semaine correspondante
            week_index = (weeks * 7 - 1 - delta_days) // 7
            day_index = candle_dt.weekday()  # 0 = lundi, 6 = dimanche
            hour_index = candle_dt.hour
            data_structure[week_index][day_index][hour_index] = pct_change

            rows = []
            for week_data in data_structure:
                for day_data in week_data:
                    rows.append(day_data)

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def data_modify(file1, file2, entropy=False, folder=DATA_DIR):
        """
        Rendre les données exploitables pour les calculs de corrélation 
        et de TE
        """
        
        def clean_list(lst, folder):
            """
            Supprimer les valeurs NaN et les matrices de manière à simplifier les calculs
            """
            
            feather = DataManagerCrypto.read_from_feather(lst, folder)
            tableau = feather.values.tolist() 
            initial_size = len(tableau)
            
            for _ in range(initial_size):
                # Supprimer les NaN si l'option est activée
                for i in range(len(tableau[0]) - 1, -1, -1):
                    if isinstance(tableau[0][i], float) and math.isnan(tableau[0][i]):
                        del tableau[0][i]
                
                # Déplacer la première sous-liste à la fin de manière à supprimer la matrice
                tableau.extend(tableau.pop(0))
            return tableau
        
        
        list_1 = clean_list(file1, folder)
        list_2 = clean_list(file2, folder)
        
        
        minimum_len = min(len(list_1), len(list_2))
        print(f"Echantillons de taille : {minimum_len}")
        # On ajuste les échantillons de manière à égaliser leur taille
        list_1 = list_1[:minimum_len]
        list_2 = list_2[:minimum_len]
        
        if entropy:
            minimum = min(min(list_1), min(list_2))
            for i in range(minimum_len):
                list_1[i] = list_1[i] - minimum
                list_2[i] = list_2[i] - minimum
        
        return list_1, list_2


    def save_to_feather(self, df, symbol, folder=DATA_DIR):
        """
        Sauvegarde la DataFrame au format Feather.
        Le fichier est nommé <symbol>_data.feather.
        """
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"{symbol}_data.feather")
        feather.write_feather(df, filename)
        print(f"Les données de {symbol} ont été sauvegardées dans {filename}")

    @staticmethod
    def read_from_feather(symbol, folder=DATA_DIR):
        """
        Lit les données depuis le fichier Feather et retourne une DataFrame.
        """
        filename = os.path.join(folder, f"{symbol}_data.feather")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} n'existe pas.")
        df = feather.read_feather(filename)
        return df

    
