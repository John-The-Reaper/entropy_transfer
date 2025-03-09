from data_manager import DataManagerCrypto
from test_stats import StatisticalAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import datetime

# ======= PARAMÈTRES DE L'ANALYSE (À MODIFIER SELON VOS BESOINS) =======
# Timeframe pour l'analyse (ex: 1d, 4h, 1h, 15m)
TIMEFRAME = "1h"

# Dates de début et de fin au format JJ-MM-AAAA
START_DATE = "01-01-2022"
END_DATE = "01-01-2025"  #datetime.now().strftime("%d-%m-%Y")  # Date actuelle par défaut

# Nombre de paires à analyser (max 50)
TOP_PAIRS = 50

# Nombre de processus parallèles à utiliser (None = auto)
# Augmentez cette valeur si vous avez un processeur puissant (8-16 cœurs)
# Diminuez-la si vous avez un processeur moins puissant ou si vous voulez
# utiliser votre ordinateur pendant l'analyse
MAX_WORKERS = None  # None = utiliser tous les cœurs disponibles

# Utiliser le cache pour les résultats de cointégration
USE_CACHE = True

# Nombre de tentatives en cas d'échec du téléchargement
MAX_RETRIES = 3

# Liste des principales cryptomonnaies (par capitalisation boursière)
CRYPTO_PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT", "MATIC/USDT:USDT",
    "LINK/USDT:USDT", "TRX/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT", "ATOM/USDT:USDT",
    "ETC/USDT:USDT", "FIL/USDT:USDT", "XLM/USDT:USDT", "XMR/USDT:USDT", "VET/USDT:USDT",
    "ALGO/USDT:USDT", "FLOW/USDT:USDT", "SAND/USDT:USDT", "FTM/USDT:USDT", "EGLD/USDT:USDT",
    "QNT/USDT:USDT", "GALA/USDT:USDT", "NEO/USDT:USDT", "EOS/USDT:USDT", "CRO/USDT:USDT",
    "THETA/USDT:USDT", "XTZ/USDT:USDT", "MANA/USDT:USDT", "GRT/USDT:USDT", "SNX/USDT:USDT",
    "ZIL/USDT:USDT", "OP/USDT:USDT", "WOO/USDT:USDT", "PEOPLE/USDT:USDT", "DYDX/USDT:USDT"
]

# ======================================================================

def sanitize_filename(name):
    """
    Convertit une chaîne en nom de fichier valide en remplaçant les caractères spéciaux.
    Fonctionne sous Windows, macOS et Linux.
    """
    # Remplacer les caractères interdits dans les noms de fichiers
    invalid_chars = r'[\\/*?:"<>|]'
    return re.sub(invalid_chars, "_", name)

def parse_date(date_str):
    """Convertit une chaîne de date au format JJ-MM-AAAA en timestamp milliseconds"""
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError(f"Format de date invalide: {date_str}. Utilisez le format JJ-MM-AAAA (ex: 01-01-2022)")

def main():
    # Conversion des dates en timestamps
    try:
        start_date = parse_date(START_DATE)
        end_date = parse_date(END_DATE)
    except ValueError as e:
        print(f"Erreur: {e}")
        return
    
    # Vérification que la date de début est antérieure à la date de fin
    if start_date >= end_date:
        print("Erreur: La date de début doit être antérieure à la date de fin")
        return
    
    # Timeframe pour l'analyse
    timeframe = TIMEFRAME
    
    # Limiter au nombre demandé
    top_cryptos = CRYPTO_PAIRS[:min(TOP_PAIRS, len(CRYPTO_PAIRS))]
    
    # Affichage des paramètres
    print(f"Paramètres de l'analyse:")
    print(f"  Timeframe: {timeframe}")
    print(f"  Date de début: {START_DATE}")
    print(f"  Date de fin: {END_DATE}")
    print(f"  Nombre de paires: {len(top_cryptos)}")
    print(f"  Utilisation du cache: {'Oui' if USE_CACHE else 'Non'}")
    print(f"  Tentatives max: {MAX_RETRIES}")
    print(f"  Processus parallèles: {'Auto' if MAX_WORKERS is None else MAX_WORKERS}")
    
    # Nettoyer le nom du timeframe pour le dossier de résultats
    safe_timeframe = sanitize_filename(timeframe)
    
    # Dossier pour stocker les résultats
    results_folder = f"cointegration_results_{safe_timeframe}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Effectuer l'analyse de cointégration
    results_df = StatisticalAnalysis.perform_cointegration_analysis(
        symbols=top_cryptos, 
        timeframe=timeframe,
        results_folder=results_folder,
        start_date=start_date,
        end_date=end_date,
        max_workers=MAX_WORKERS,
        use_cache=USE_CACHE,
        max_retries=MAX_RETRIES
    )
    
    # Vérifier si des résultats ont été obtenus
    if results_df.empty:
        print("\nAucun résultat de cointégration n'a été obtenu. Impossible de continuer l'analyse.")
        return
    
    # Afficher les 10 meilleures paires cointégrées
    print("\nLes 10 meilleures paires cointégrées:")
    if 'cointegrated' in results_df.columns:
        top_10 = results_df[results_df['cointegrated'] == True].head(10)
        if not top_10.empty:
            print(top_10[['pair', 'p_value', 'test_statistic', 'beta']])
            
            # Créer une heatmap de cointégration
            create_cointegration_heatmap(results_df, top_cryptos, results_folder, timeframe)
            
            # Visualiser les séries pour les 3 meilleures paires
            if len(top_10) >= 3:
                for i in range(min(3, len(top_10))):
                    pair_data = top_10.iloc[i]
                    symbol1 = pair_data['symbol1']
                    symbol2 = pair_data['symbol2']
                    plot_cointegrated_pair(symbol1, symbol2, results_folder, timeframe, start_date, end_date)
        else:
            print("Aucune paire cointégrée n'a été trouvée.")
    else:
        print("Le format des résultats est incorrect. La colonne 'cointegrated' est manquante.")
    
    print("\nAnalyse de cointégration terminée!")

def create_cointegration_heatmap(results_df, symbols, results_folder, timeframe):
    """Crée une heatmap des p-values de cointégration"""
    # Initialiser une matrice vide
    n = len(symbols)
    matrix = pd.DataFrame(index=symbols, columns=symbols)
    
    # Remplir la matrice avec les p-values
    for _, row in results_df.iterrows():
        symbol1 = row['symbol1']
        symbol2 = row['symbol2']
        p_value = row['p_value']
        matrix.loc[symbol1, symbol2] = p_value
        matrix.loc[symbol2, symbol1] = p_value  # Symétrique
    
    # Remplir la diagonale avec des 1 (pas de cointégration avec soi-même)
    for symbol in symbols:
        matrix.loc[symbol, symbol] = 1.0
    
    # Convertir en valeurs numériques
    matrix = matrix.astype(float)
    
    # Créer la heatmap
    plt.figure(figsize=(14, 12))
    mask = matrix.isnull()
    
    # Utiliser une échelle de couleur où les valeurs faibles (forte cointégration) sont en vert
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    # Créer la heatmap
    sns.heatmap(matrix, mask=mask, cmap=cmap, vmax=0.05, vmin=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5},
                xticklabels=True, yticklabels=True)
    
    plt.title(f'Heatmap de Cointégration (p-values) - {timeframe}', fontsize=16)
    plt.tight_layout()
    
    # Sauvegarder la figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Nettoyer le nom du timeframe
    safe_timeframe = sanitize_filename(timeframe)
    
    heatmap_file = os.path.join(results_folder, f'cointegration_heatmap_{safe_timeframe}_{timestamp}.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap sauvegardée dans {heatmap_file}")

def plot_cointegrated_pair(symbol1, symbol2, results_folder, timeframe, start_date, end_date):
    """Visualise une paire cointégrée"""
    # Charger les données
    dm = DataManagerCrypto()
    
    # Récupérer les données
    candles1 = dm.fetch_historical_data(symbol1, timeframe, start_date, end_date)
    candles2 = dm.fetch_historical_data(symbol2, timeframe, start_date, end_date)
    
    # Vérifier si des données ont été récupérées
    if not candles1 or not candles2:
        print(f"Données insuffisantes pour visualiser {symbol1} et {symbol2}")
        return
    
    # Convertir en DataFrame
    df1 = pd.DataFrame(candles1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df2 = pd.DataFrame(candles2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')
    
    df1.set_index('timestamp', inplace=True)
    df2.set_index('timestamp', inplace=True)
    
    # S'assurer que les deux séries ont la même longueur
    common_index = df1.index.intersection(df2.index)
    if len(common_index) < 30:  # Au moins 30 points de données communs
        print(f"Données communes insuffisantes pour visualiser {symbol1} et {symbol2}")
        return
        
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    
    # Normaliser les prix pour une meilleure visualisation
    df1_norm = df1['close'] / df1['close'].iloc[0]
    df2_norm = df2['close'] / df2['close'].iloc[0]
    
    # Créer le graphique
    plt.figure(figsize=(12, 6))
    plt.plot(df1_norm.index, df1_norm, label=symbol1)
    plt.plot(df2_norm.index, df2_norm, label=symbol2)
    plt.title(f'Paire cointégrée: {symbol1} et {symbol2} ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Prix normalisé')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder la figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Nettoyer les noms de symboles
    safe_symbol1 = sanitize_filename(symbol1)
    safe_symbol2 = sanitize_filename(symbol2)
    safe_timeframe = sanitize_filename(timeframe)
    
    pair_file = os.path.join(results_folder, f'pair_{safe_symbol1}_{safe_symbol2}_{safe_timeframe}_{timestamp}.png')
    plt.savefig(pair_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graphique pour {symbol1} et {symbol2} sauvegardé dans {pair_file}")

if __name__ == "__main__":
    main()
