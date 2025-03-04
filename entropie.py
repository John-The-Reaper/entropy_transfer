import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from stats_test import transfer_entropy
from data_manager import DataManagerCrypto
from stats_test import StatisticalAnalysis

LAG_RANGE = 300  # Nombre maximal de lags à tester
SERIES_SOURCE = "BTCUSDT"
SERIE_TARGET = "ETHUSDT"
WINDOW_SIZE = 100

###############################################################################
# Affichage des résultats
###############################################################################


def analyze_te_by_lag(series_source, series_target):
    te_values = StatisticalAnalysis.transfer_entropy(series_source, series_target, LAG_RANGE)

    max_val = max(te_values, key=lambda item: item[1])
    print("Valeur maximale du TE :", max_val)

    lags, values = zip(*te_values)
    mean_te = np.mean(values)
    window_size = max(10, len(values) // 8)
    moving_avg = StatisticalAnalysis.moving_average(values, window_size)

    return lags, values, moving_avg, mean_te

def plot_te_evolution(te_values):
    plt.figure(figsize=(12, 6)) 
    sns.set_style("darkgrid") 

    plt.plot(te_values, linestyle='-', markersize = 3, linewidth=1.5, color='royalblue', alpha=0.7, label="Transfert d'Entropie")
    plt.gca().set_facecolor('#f4f4f4')  
    
    plt.xlabel("Temps (en periodes)", fontsize=12, fontweight='bold')
    plt.ylabel("TE", fontsize=12, fontweight='bold')
    plt.title("Évolution du Transfert d'Entropie", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show(block=False)  # Affichage en mode non bloquant


def plot_te_by_lag(lags, values, moving_avg, mean_te):
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")  
    
    plt.plot(lags, values, linewidth=2, linestyle='-', color='royalblue', alpha=0.7, label="TE par lag")
    plt.plot(lags, moving_avg, linestyle='-', linewidth=1, color='red', alpha=0.8, label=f"Moyenne mobile ({len(moving_avg)} points)")
    plt.axhline(y=mean_te, color='green', linestyle='dotted', linewidth=2, label=f"Moyenne globale ({mean_te:.4f})")
    
    plt.xlabel("Lag (en nombre de périodes)", fontsize=12, fontweight='bold')
    plt.ylabel("TE", fontsize=12, fontweight='bold')
    plt.title("Transfert d'Entropie en fonction du lag", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show(block=False)  # Affichage en mode non bloquant

###############################################################################
# Exécution principale
###############################################################################

def main():
    
    te_values = StatisticalAnalysis.compute_te_over_time(SERIES_SOURCE, SERIE_TARGET, WINDOW_SIZE)
    lags, values, moving_avg, mean_te = analyze_te_by_lag(SERIES_SOURCE, SERIE_TARGET)

    #  Affichage des graphes simultanément
    plot_te_evolution(te_values)  
    plot_te_by_lag(lags, values, moving_avg, mean_te)  
    
    input("Appuyez sur Entrée pour fermer les graphiques...")  # Attente pour éviter la fermeture immédiate

if __name__ == '__main__':
    main()
