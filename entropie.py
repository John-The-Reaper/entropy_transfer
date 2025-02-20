import numpy as np
from pyinform.transferentropy import transfer_entropy
import matplotlib.pyplot as plt
from stats_test import StatisticalAnalysis


def main():
    # Tester plusieurs décalages
    te_values = StatisticalAnalysis.transfer_entropy("DYMUSDT", "BTCUSDT", 300)
    lags_range = 200

    min_val = min(te_values, key=lambda item: item[1])  # Trouve la liste avec la plus petite valeur en index 1
    print("Valeur la plus intéressante : ", min_val)

    lags, values = zip(*te_values)

    plt.plot(lags, values, marker='o', linestyle='-')
    plt.xlabel("Lag (en nombre de periodes)")
    plt.ylabel("Transfert d'entropie")
    plt.title("Évolution du TE en fonction du lag")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
