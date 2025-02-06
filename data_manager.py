import ccxt
import os
from datetime import datetime, timedelta
from time import sleep
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from tqdm import tqdm



###############################################################################
# CLASSE DE GESTION DES DONNÉES
###############################################################################

INTERVALS_PER_DAY = 24 // 4      # 6 intervalles par jour
DATA_DIR = "data"              # Répertoire de sauvegarde des Feather




class DataManager:
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
          - Week, Day, puis les valeurs de chaque intervalle (H0, H4, …, H20)
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
            hour_index = candle_dt.hour // 4
            data_structure[week_index][day_index][hour_index] = pct_change
            
            rows = []
            for week_data in data_structure:
                for day_data in week_data:
                    rows.append(day_data)

        df = pd.DataFrame(rows)
        return df

    
    def save_to_feather(self, df, symbol, folder=DATA_DIR):
        """
        Sauvegarde la DataFrame au format Feather.
        Le fichier est nommé <symbol>_data.feather.
        """
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"{symbol}_data.feather")
        feather.write_feather(df, filename)
        print(f"Les données de {symbol} ont été sauvegardées dans {filename}")

    def read_from_feather(self, symbol, folder=DATA_DIR):
        """
        Lit les données depuis le fichier Feather et retourne une DataFrame.
        """
        filename = os.path.join(folder, f"{symbol}_data.feather")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} n'existe pas.")
        df = feather.read_feather(filename)
        return df
    