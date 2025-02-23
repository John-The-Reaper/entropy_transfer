import numpy as np
import matplotlib.pyplot as plt

from stats_test import StatisticalAnalysis

###############################################################################
# Test TE avec affichage graphique
###############################################################################

LAG_RANGE = 300

def main():

    te_values = StatisticalAnalysis.transfer_entropy("BTCUSDT", "ETHUSDT", LAG_RANGE)

    max_val = max(te_values, key=lambda item: item[1])  # Trouve la liste avec la plus grande valeur en index 1
    print("Valeur la plus intéressante : ", max_val)

    lags, values = zip(*te_values)

    plt.plot(lags, values, marker='o', linestyle='-')
    plt.xlabel("Lag (en nombre de periodes)")
    plt.ylabel("Transfert d'entropie")
    plt.title("Évolution du TE en fonction du lag")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()