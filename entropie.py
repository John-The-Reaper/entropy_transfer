import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from stats_test import StatisticalAnalysis

###############################################################################
# Test TE avec affichage graphique
###############################################################################

LAG_RANGE = 300

def main():
    te_values = StatisticalAnalysis.transfer_entropy("ETHUSDT", "INJUSDT", LAG_RANGE)
    
    max_val = max(te_values, key=lambda item: item[1])
    print("Valeur la plus intéressante : ", max_val)

    # Extraction des lags et des valeurs TE
    lags, values = zip(*te_values)

    # Moyenne globale du TE
    mean_te = np.mean(values)
    print(f"Moyenne globale du transfert d'entropie : {mean_te:.4f}")

    # Fenêtre de lissage ajustée pour un rendu plus fluide
    window_size = max(10, len(values) // 8)

    # Calcul de la moyenne mobile complète
    moving_avg = StatisticalAnalysis.moving_average(values, window_size)

    # 🔹 Amélioration du design du graphique
    plt.figure(figsize=(12, 6))  # Taille du graphique
    sns.set_style("darkgrid") # Applique le style Seaborn

    # Courbe principale
    plt.plot(lags, values, marker='o', markersize=3, linestyle='-', color='royalblue', alpha=0.7, label="TE par lag")

    # Moyenne mobile
    plt.plot(lags, moving_avg, linestyle='-', linewidth=2, color='red', alpha=0.8, label=f"Moyenne mobile ({window_size} points)")

    # Moyenne globale
    plt.axhline(y=mean_te, color='green', linestyle='dotted', linewidth=2, label=f"Moyenne globale ({mean_te:.4f})")

    # Personnalisation des axes
    plt.xlabel("Lag (en nombre de périodes)", fontsize=12, fontweight='bold')
    plt.ylabel("Transfert d'entropie", fontsize=12, fontweight='bold')
    plt.title("Évolution du TE en fonction du lag", fontsize=14, fontweight='bold')

    # Ajout d'une légende améliorée
    plt.legend(fontsize=10, loc='upper right')

    # Ajout d'une grille plus légère
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Affichage
    plt.show()


if __name__ == '__main__':
    main()
