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
    
    def fetch_historical_data(self, symbol, timeframe, start_ts, end_ts, use_cache=True):
        """
        Récupère les chandeliers depuis start_ts jusqu'à end_ts (en millisecondes)
        en paginant si nécessaire. Utilise des fichiers de cache feather si disponibles.
        
        Args:
            symbol: Symbole de la paire (ex: "BTC/USDT")
            timeframe: Intervalle de temps (ex: "1d", "1h", "15m")
            start_ts: Timestamp de début en millisecondes ou date au format "JJ-MM-AAAA"
            end_ts: Timestamp de fin en millisecondes ou date au format "JJ-MM-AAAA"
            use_cache: Si True, utilise le cache feather si disponible
            
        Returns:
            Liste de chandeliers OHLCV
        """
        # Conversion des timestamps en dates pour l'affichage
        if isinstance(start_ts, str):
            try:
                start_dt = datetime.strptime(start_ts, "%d-%m-%Y")
                start_ts = int(start_dt.timestamp() * 1000)
            except ValueError as e:
                print(f"Erreur de format de date pour start_ts: {e}")
                return []
            
        elif isinstance(start_ts, int) and start_ts < 10000000000:  # Si c'est un timestamp en secondes
            start_ts *= 1000
            
        if isinstance(end_ts, str):
            try:
                end_dt = datetime.strptime(end_ts, "%d-%m-%Y")
                end_ts = int(end_dt.timestamp() * 1000)
            except ValueError as e:
                print(f"Erreur de format de date pour end_ts: {e}")
                return []
        elif isinstance(end_ts, int) and end_ts < 10000000000:  # Si c'est un timestamp en secondes
            end_ts *= 1000

        start_date = datetime.fromtimestamp(start_ts / 1000).strftime("%d-%m-%Y")
        end_date = datetime.fromtimestamp(end_ts / 1000).strftime("%d-%m-%Y")
        
        # Remplacer les caractères spéciaux dans le nom du symbole pour le nom de fichier
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        safe_timeframe = timeframe.replace("/", "_")
        cache_dir = os.path.join(DATA_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Nom du fichier de cache
        cache_file = os.path.join(cache_dir, f"{safe_symbol}_{safe_timeframe}_cache.feather")
        
        # Vérifier si le fichier de cache existe
        cached_data = None
        start_ts_to_fetch = start_ts
        end_ts_to_fetch = end_ts
        
        if use_cache and os.path.exists(cache_file):
            try:
                # Charger les données du cache
                df_cache = feather.read_feather(cache_file)
                
                # Convertir en timestamp si nécessaire
                if 'timestamp' in df_cache.columns and not pd.api.types.is_numeric_dtype(df_cache['timestamp']):
                    df_cache['timestamp'] = df_cache['timestamp'].astype(np.int64) // 10**6
                
                # Si le cache contient des données
                if not df_cache.empty:
                    # Déterminer la plage de dates dans le cache
                    min_ts = df_cache['timestamp'].min()
                    max_ts = df_cache['timestamp'].max()
                    
                    print(f"Cache trouvé pour {symbol} ({timeframe}) couvrant du {datetime.fromtimestamp(min_ts / 1000).strftime('%d-%m-%Y')} au {datetime.fromtimestamp(max_ts / 1000).strftime('%d-%m-%Y')}")
                    
                    # Filtrer les données dans la plage demandée
                    df_filtered = df_cache[(df_cache['timestamp'] >= start_ts) & (df_cache['timestamp'] <= end_ts)]
                    
                    # Convertir en liste de chandeliers
                    cached_data = df_filtered.values.tolist()
                    
                    # Déterminer quelles données manquent
                    if min_ts <= start_ts and max_ts >= end_ts:
                        # Le cache couvre toute la plage demandée
                        print(f"Utilisation du cache pour {symbol} ({timeframe}) - données complètes")
                        return cached_data
                    elif min_ts <= start_ts:
                        # Le cache couvre le début mais pas la fin
                        start_ts_to_fetch = max_ts + 1
                        print(f"Téléchargement des données manquantes pour {symbol} ({timeframe}) du {datetime.fromtimestamp(start_ts_to_fetch / 1000).strftime('%d-%m-%Y')} au {end_date}")
                    elif max_ts >= end_ts:
                        # Le cache couvre la fin mais pas le début
                        end_ts_to_fetch = min_ts - 1
                        print(f"Téléchargement des données manquantes pour {symbol} ({timeframe}) du {start_date} au {datetime.fromtimestamp(end_ts_to_fetch / 1000).strftime('%d-%m-%Y')}")
                    else:
                        # Le cache ne couvre pas la plage demandée ou il y a un trou
                        # On télécharge tout et on fusionnera plus tard
                        print(f"Téléchargement des données manquantes pour {symbol} ({timeframe}) du {start_date} au {end_date}")
            except Exception as e:
                print(f"Erreur lors de la lecture du cache pour {symbol} ({timeframe}): {e}")
                cached_data = None
        
        # Si aucune donnée dans le cache ou si des données manquent, télécharger les données
        new_candles = []
        if cached_data is None or start_ts_to_fetch <= end_ts_to_fetch:
            # Création d'une barre de progression colorée avec informations de dates
            pbar = tqdm(
                desc=f"Téléchargement {symbol} ({timeframe})", 
                unit="données",
                bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            pbar.colour = "green"  # Définir la couleur de la barre
            
            since = start_ts_to_fetch
            limit = 1000  # limite de chandeliers par appai
            
            while since <= end_ts_to_fetch:
                try:
                    candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                except Exception as e:
                    print(f"Erreur lors de la récupération de {symbol}: {e}")
                    break

                if not candles:
                    break

                valid_candles = []
                for candle in candles:
                    ts = candle[0]
                    if ts > end_ts_to_fetch:
                        # On arrête si l'on dépasse la date de fin
                        break
                    valid_candles.append(candle)
                    
                if valid_candles:
                    new_candles.extend(valid_candles)
                    first_ts = valid_candles[0][0]
                    last_ts = valid_candles[-1][0]
                    current_date = datetime.fromtimestamp(last_ts / 1000).strftime("%d-%m-%Y")
                    
                    # Mettre à jour la description de la barre avec la date actuelle
                    fetch_start_date = datetime.fromtimestamp(start_ts_to_fetch / 1000).strftime("%d-%m-%Y")
                    fetch_end_date = datetime.fromtimestamp(end_ts_to_fetch / 1000).strftime("%d-%m-%Y")
                    pbar.set_description(f"Téléchargement {symbol} ({timeframe}): {fetch_start_date} -> {current_date} -> {fetch_end_date}")
                    pbar.update(len(valid_candles))

                if not candles or candles[-1][0] >= end_ts_to_fetch:
                    break
                    
                since = candles[-1][0] + 1
                sleep(self.exchange.rateLimit / 1000)
                
            pbar.close()
        
        # Fusionner les données téléchargées avec les données en cache
        all_candles = []
        
        if cached_data is not None:
            # Convertir les données en DataFrame pour la fusion
            df_cached = pd.DataFrame(cached_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if new_candles:
                df_new = pd.DataFrame(new_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Fusionner les DataFrames
                df_combined = pd.concat([df_cached, df_new], ignore_index=True)
                
                # Supprimer les doublons potentiels
                df_combined = df_combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # Convertir en liste de candles
                all_candles = df_combined.values.tolist()
                
                # Mettre à jour le cache
                self.update_candles_cache(df_combined, symbol, timeframe)
            else:
                all_candles = cached_data
        else:
            if new_candles:
                all_candles = new_candles
                
                # Convertir en DataFrame pour la mise en cache
                df_new = pd.DataFrame(new_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Mettre à jour le cache
                self.update_candles_cache(df_new, symbol, timeframe)
        
        # Filtrer une dernière fois pour s'assurer que seules les données dans la plage demandée sont renvoyées
        filtered_candles = [candle for candle in all_candles if start_ts <= candle[0] <= end_ts]
        
        return filtered_candles
        
    def update_candles_cache(self, df, symbol, timeframe):
        """Met à jour le fichier de cache pour une paire et un timeframe donnés"""
        # Remplacer les caractères spéciaux dans le nom du symbole pour le nom de fichier
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        safe_timeframe = timeframe.replace("/", "_")
        cache_dir = os.path.join(DATA_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Nom du fichier de cache
        cache_file = os.path.join(cache_dir, f"{safe_symbol}_{safe_timeframe}_cache.feather")
        
        # Sauvegarder les données
        feather.write_feather(df, cache_file)
        print(f"Cache mis à jour pour {symbol} ({timeframe})")
        
    def save_to_feather(self, df, symbol, folder=DATA_DIR, timeframe=None):
        """
        Sauvegarde la DataFrame au format Feather.
        Le fichier est nommé <symbol>_<timeframe>_data.feather si timeframe est spécifié.
        """
        os.makedirs(folder, exist_ok=True)
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        
        if timeframe:
            safe_timeframe = timeframe.replace("/", "_")
            filename = os.path.join(folder, f"{safe_symbol}_{safe_timeframe}_data.feather")
        else:
            filename = os.path.join(folder, f"{safe_symbol}_data.feather")
            
        feather.write_feather(df, filename)
        print(f"Les données de {symbol} ont été sauvegardées dans {filename}")

    @staticmethod
    def read_from_feather(symbol, folder=DATA_DIR, timeframe=None):
        """
        Lit les données depuis le fichier Feather et retourne une DataFrame.
        """
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        
        if timeframe:
            safe_timeframe = timeframe.replace("/", "_")
            filename = os.path.join(folder, f"{safe_symbol}_{safe_timeframe}_data.feather")
        else:
            filename = os.path.join(folder, f"{safe_symbol}_data.feather")
            
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} n'existe pas.")
            
        df = feather.read_feather(filename)
        return df

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

    