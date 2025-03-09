from data_manager import DataManagerCrypto
from test_stats import StatisticalAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse

def parse_date(date_str):
    """Convertit une chaîne de date au format JJ-MM-AAAA en timestamp milliseconds"""
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError(f"Format de date invalide: {date_str}. Utilisez le format JJ-MM-AAAA (ex: 01-01-2022)")

def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Test de cointégration entre paires de cryptomonnaies")
    parser.add_argument("--timeframe", type=str, default="1d", 
                        help="Timeframe pour les données (ex: 1d, 4h, 1h, 15m)")
    parser.add_argument("--start_date", type=str, default="01-01-2022", 
                        help="Date de début au format JJ-MM-AAAA (ex: 01-01-2022)")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%d-%m-%Y"), 
                        help="Date de fin au format JJ-MM-AAAA (ex: 31-12-2022)")
    parser.add_argument("--pairs", type=str, nargs="+", 
                        default=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
                        help="Liste des paires à analyser (ex: BTC/USDT ETH/USDT)")
    
    args = parser.parse_args()
    
    # Conversion des dates en timestamps
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    except ValueError as e:
        print(f"Erreur: {e}")
        return
    
    # Vérification que la date de début est antérieure à la date de fin
    if start_date >= end_date:
        print("Erreur: La date de début doit être antérieure à la date de fin")
        return
    
    # Liste des cryptomonnaies à tester
    test_cryptos = args.pairs
    timeframe = args.timeframe
    
    # Affichage des paramètres
    print(f"Paramètres de l'analyse:")
    print(f"  Timeframe: {timeframe}")
    print(f"  Date de début: {args.start_date}")
    print(f"  Date de fin: {args.end_date}")
    print(f"  Paires: {', '.join(test_cryptos)}")
    
    # Dossier pour stocker les résultats
    results_folder = f"cointegration_results_{timeframe}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Initialisation du gestionnaire de données
    manager = DataManagerCrypto()
    
    print(f"Téléchargement des données pour {len(test_cryptos)} paires...")
    
    # Dictionnaire pour stocker les DataFrames
    dataframes = {}
    
    # Télécharger les données pour chaque symbole
    for symbol in test_cryptos:
        try:
            print(f"Téléchargement de {symbol} en {timeframe}...")
            candles = manager.fetch_historical_data(symbol, timeframe, start_date, end_date)
            
            if not candles or len(candles) < 30:
                print(f"Données insuffisantes pour {symbol}, ignoré.")
                continue
                
            # Conversion en DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Stockage dans le dictionnaire
            dataframes[symbol] = df
            print(f"Téléchargement réussi pour {symbol}: {len(df)} points de données")
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
    
    # Tester la cointégration pour toutes les paires
    if len(dataframes) >= 2:
        symbols = list(dataframes.keys())
        
        # Créer un DataFrame pour stocker les résultats
        results = []
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1 = symbols[i]
                symbol2 = symbols[j]
                
                print(f"\nTest de cointégration entre {symbol1} et {symbol2}...")
                
                df1 = dataframes[symbol1]['close']
                df2 = dataframes[symbol2]['close']
                
                # S'assurer que les deux séries ont la même longueur
                common_index = df1.index.intersection(df2.index)
                if len(common_index) < 30:
                    print(f"Données communes insuffisantes pour {symbol1} et {symbol2}, ignoré.")
                    continue
                    
                df1 = df1.loc[common_index]
                df2 = df2.loc[common_index]
                
                try:
                    coint_result = StatisticalAnalysis.test_cointegration(df1, df2)
                    
                    # Ajouter les résultats au DataFrame
                    result = {
                        'pair': f"{symbol1}_{symbol2}",
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'cointegrated': coint_result['cointegrated'],
                        'p_value': coint_result['p_value'],
                        'test_statistic': coint_result['test_statistic'],
                        'beta': coint_result['beta']
                    }
                    results.append(result)
                    
                    print(f"Résultat du test de cointégration:")
                    print(f"  Cointégré: {coint_result['cointegrated']}")
                    print(f"  P-value: {coint_result['p_value']}")
                    print(f"  Statistique de test: {coint_result['test_statistic']}")
                    print(f"  Beta: {coint_result['beta']}")
                    
                    # Si les séries sont cointégrées, créer un graphique
                    if coint_result['cointegrated']:
                        plt.figure(figsize=(12, 6))
                        
                        # Normaliser les prix pour une meilleure visualisation
                        df1_norm = df1 / df1.iloc[0]
                        df2_norm = df2 / df2.iloc[0]
                        
                        plt.plot(df1_norm.index, df1_norm, label=symbol1)
                        plt.plot(df2_norm.index, df2_norm, label=symbol2)
                        plt.title(f'Paire cointégrée: {symbol1} et {symbol2} ({timeframe})')
                        plt.xlabel('Date')
                        plt.ylabel('Prix normalisé')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Sauvegarder la figure
                        pair_file = os.path.join(results_folder, f'pair_{symbol1.replace("/", "_")}_{symbol2.replace("/", "_")}.png')
                        plt.savefig(pair_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Graphique sauvegardé dans {pair_file}")
                except Exception as e:
                    print(f"Erreur lors du test de cointégration pour {symbol1} et {symbol2}: {e}")
        
        # Sauvegarder les résultats dans un fichier CSV
        if results:
            results_df = pd.DataFrame(results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(results_folder, f'cointegration_results_{timeframe}_{timestamp}.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\nRésultats sauvegardés dans {results_file}")
            
            # Afficher un résumé
            print(f"\nRésumé des résultats:")
            print(f"  Nombre total de paires testées: {len(results)}")
            print(f"  Nombre de paires cointégrées: {results_df['cointegrated'].sum()}")
            
            # Afficher les paires cointégrées triées par p-value
            if results_df['cointegrated'].sum() > 0:
                print("\nPaires cointégrées (triées par p-value):")
                cointegrated_pairs = results_df[results_df['cointegrated'] == True].sort_values('p_value')
                for _, row in cointegrated_pairs.iterrows():
                    print(f"  {row['symbol1']} - {row['symbol2']}: p-value = {row['p_value']:.6f}, beta = {row['beta']:.6f}")
        else:
            print("Aucun résultat de cointégration n'a été obtenu.")
    else:
        print("Pas assez de données pour effectuer des tests de cointégration.")

if __name__ == "__main__":
    main() 