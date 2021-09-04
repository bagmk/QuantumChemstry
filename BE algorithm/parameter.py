"""
Zu verwendende Parameter.

Die hier angegebenen Parameter werden von den Funktionen verwendet, falls keine anderen Festlegungen in dem
Hauptskript vorgenommen wurden.

Copyright (c) 2018 Daniel Weber.
Distributed under the MIT License.
(See accompanying LICENSE file or copy at https://github.com/dnlwbr/Born-Oppenheimer-Approximation/blob/master/LICENSE)
"""

import numpy as np


# adiabatischer Parameter
eps = 1/8

# reduziertes Planck'sches Wirkungsquantum
h_red = 1

# Auswertungsbereich
Nx = 2 ** 9                                      # Anzahl der Gitterpunkte in x-Richtung
Ny = 2 ** 8                                      # Anzahl der Gitterpunkte in y-Richtung
xrange = 5                                       # Intervallgrenze x-Richtung
yrange = 4                                       # Intervallgrenze y-Richtung
x = np.arange(-xrange, xrange, 2 * xrange / Nx)  # Intervall x-Richtung
y = np.arange(-yrange, yrange, 2 * yrange / Ny)  # Intervall y-Richtung

# Anfangswerte
px = 1		                                     # Anfangsimpuls in x-Richtung

# Teilchenmasse
m = 1

# Schrittweite im Strang-Splitting
h = 1/60000
# Ist ein Zeitpunkt nicht erreichbar, wird automatisch die nächstmögliche kleinere Schrittweite verwendet
