"""
Hauptprogramm.

Copyright (c) 2018 Daniel Weber.
Distributed under the MIT License.
(See accompanying LICENSE file or copy at https://github.com/dnlwbr/Born-Oppenheimer-Approximation/blob/master/LICENSE)
"""

import numpy as np
from parameter import eps, xrange, yrange, x, y
from function_data_base import potential, plot_potential, plot_potentialHSL, plot_potential3D, plot_potential_wave,\
    plot_HSL_wave, potential_VBH, correction_term, plot_potential_VBH, plot_correction_term,\
    multiple_propagation_times, compare_grid, plot_grid, compare_splitting_steps, plot_splitting, compare_error_eps,\
    plot_error_eps, compare_error_time, plot_error_time, compare_error_time_VMU, plot_error_time_VMU, ask_data_handling


retries = 3
while True:

    todo = str(input("""
    Was soll berechnet werden?
         [1] Konturplot des Potentials
         [2] HSL-Plot des Potentials
         [3] 3D-Plot des Potentials
         [4] Konturplot der Wellenfunktion im Potential
         [5] HSL-Plot der Wellenfunktion
         [6] Plot Born-Huang-Potential
         [7] Plot Korrekturterm
         [8] Konvergenztest - Gittergröße
         [9] Konvergenztest - Zeitschrittweite beim Strang-Splitting
        [10] Qualitätstest nach epsilon
        [11] Qualitätstest nach Zeit
        [12] Qualitätstest nach Zeit (nur 2. Ordnung)
         [Q] Beenden
    >> """))

    if todo == "1":
        # Konturplot des Potentials
        print("Potential wird berechnet...")
        V = potential(x, y)

        print("Graph wird gezeichnet...")
        plot_potential(x, y, V)
        break

    elif todo == "2":
        # HSL-Plot des Potentials
        print("Potential wird berechnet...")
        V = potential(x, y)

        print("Graph wird gezeichnet...")
        plot_potentialHSL(V, xrange, yrange)
        break

    elif todo == "3":
        # 3D-Plot des Potentials
        print("Potential wird berechnet...")
        V = potential(x, y)

        print("Graph wird gezeichnet...")
        plot_potential3D(x, y, V)
        break

    elif todo == "4":
        # Konturplot der Wellenfunktion im Potential
        print("Potential wird berechnet...")
        V = potential(x, y)

        print("Wellenfunktion wird berechnet...")
        times = [t for t in range(0, 6, 1)]
        allPsi = multiple_propagation_times(x, y, times)

        print("Graph wird gezeichnet...")
        plot_potential_wave(x, y, V, allPsi)
        break

    elif todo == "5":
        # HSL-Plot der Wellenfunktion
        print("Wellenfunktion wird berechnet...")
        times = [t for t in range(0, 6, 1)]
        allPsi = multiple_propagation_times(x, y, times)

        print("Graph wird gezeichnet...")
        plot_HSL_wave(allPsi, xrange, yrange)
        break

    elif todo == "6":
        # Plot Born-Huang-Potential
        print("Born-Huang-Potential wird berechnet...")
        VBH = potential_VBH(x)

        print("Graph wird gezeichnet...")
        plot_potential_VBH(x, VBH)
        break

    elif todo == "7":
        # Plot Korrekturterm
        print("Korrekturterm wird berechnet...")
        M = correction_term(x, eps)

        print("Graph wird gezeichnet...")
        plot_correction_term(x, M)
        break

    elif todo == "8":
        # Konvergenztest - Gittergröße
        times = np.linspace(0, 12, 25)
        nx1, nx2, nx3, nx4, nx5, ny = 6, 7, 8, 9, 10, None
        # nx1, nx2, nx3, nx4, nx5, ny = 9, 9, 9, 9, 9, [6, 7, 8, 9, 10]

        handling = ask_data_handling()
        if handling is None:
            continue

        print("\nWellenfunktionen werden berechnet...")
        normDiff = compare_grid(xrange, yrange, times, nx1, nx2, nx3, nx4, nx5, ny=ny,
                                potV=potential, effektiv=False, handling=handling)

        print("effektive Wellenfunktionen werden berechnet...")
        normDiff_eff = compare_grid(xrange, yrange, times, nx1, nx2, nx3, nx4, nx5, ny=ny)

        print("Graph wird gezeichnet...")
        plot_grid(normDiff, normDiff_eff, times, nx1, nx2, nx3, nx4, nx5, ny=ny)
        break

    elif todo == "9":
        # Konvergenztest - Zeitschrittweite beim Strang-Splitting
        times = np.linspace(0, 12, 25)
        h1, h2, h3, h4, h5 = 1/40000, 1/50000, 1/60000, 1/70000, 1/80000

        handling = ask_data_handling()
        if handling is None:
            continue

        print("\nPotential wird berechnet...")
        V = potential(x, y)

        print("Wellenfunktionen werden berechnet...")
        normDiff = compare_splitting_steps(x, y, times, h1, h2, h3, h4, h5, potV=V, effektiv=False, handling=handling)

        print("effektive Wellenfunktionen werden berechnet...")
        normDiff_eff = compare_splitting_steps(x, y, times, h1, h2, h3, h4, h5)

        print("Graph wird gezeichnet...")
        plot_splitting(normDiff, normDiff_eff, times, h1, h2, h3, h4, h5)
        break

    elif todo == "10":
        # Qualitätstest nach epsilon
        t = 3
        epsilons = [2**(-1), 2**(-1.5), 2**(-2), 2**(-2.5), 2**(-3), 2**(-3.5), 2**(-4)]

        handling = ask_data_handling()
        if handling is None:
            continue

        print("\nPotential wird berechnet...")
        V = potential(x, y)

        print("Wellenfunktionen werden berechnet...")
        normDiff_eff, normDiff_effV, normDiff_effVM, normDiff_effVMU = compare_error_eps(x, y, t, epsilons, V, handling)

        print("Graph wird gezeichnet...")
        plot_error_eps(normDiff_eff, normDiff_effV, normDiff_effVM, normDiff_effVMU, t, epsilons)
        break

    elif todo == "11":
        # Qualitätstest nach Zeit
        times = np.linspace(0, 12, 25)

        handling = ask_data_handling()
        if handling is None:
            continue

        print("\nPotential wird berechnet...")
        V = potential(x, y)

        print("Wellenfunktionen werden berechnet...")
        normDiff_eff, normDiff_effV, normDiff_effVM, normDiff_effVMU = compare_error_time(x, y, times, V, handling)

        print("Graph wird gezeichnet...")
        plot_error_time(times, normDiff_eff, normDiff_effV, normDiff_effVM, normDiff_effVMU)
        break

    elif todo == "12":
        # Qualitätstest nach Zeit (nur 2. Ordnung)
        times = np.linspace(0, 50, 101)
        epsilons = [1/4, 1/8, 1/16]

        handling = ask_data_handling()
        if handling is None:
            continue

        print("\nPotential wird berechnet...")
        V = potential(x, y)

        print("Wellenfunktionen werden berechnet...")
        normDiff_effVMU_eps = compare_error_time_VMU(x, y, times, epsilons, V, handling)

        print("Graph wird gezeichnet...")
        plot_error_time_VMU(times, normDiff_effVMU_eps, interpolate=True)
        break

    elif todo in ("Q", "q"):
        print("Beendet.")
        break

    else:
        print("Ungültige Eingabe!")

    retries -= 1
    if retries == 0:
        print("Abbruch aufgrund von drei fehlerhaften Versuchen.")
        break
