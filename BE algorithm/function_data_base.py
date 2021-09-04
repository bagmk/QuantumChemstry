"""
Sammlung von Funktionen.

Copyright (c) 2018 Daniel Weber.
Distributed under the MIT License.
(See accompanying LICENSE file or copy at https://github.com/dnlwbr/Born-Oppenheimer-Approximation/blob/master/LICENSE)
"""

import os
import pickle
import sys
import time
from colorsys import hls_to_rgb
from fractions import Fraction

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from numpy import exp, sqrt
from numpy.fft import fft, ifft, fft2, ifft2, fftfreq

from parameter import eps, h_red, px, m, h


# Frequenz
def frequency(x):
    """Berechnet die Funktionswerte der Frequenz."""
    result = sqrt(2 * (1 + x**4))
    return result


# Ableitung der Frequenz
def frequency_derived(x):
    """Berechnet die Funktionswerte der Ableitung der Frequenz."""
    result = 2 * sqrt(2) * x**3 / sqrt(1 + x ** 4)
    return result


# Anfangswert der Wellenfunktion der Kerne (x-Richtung)
def calc_psi_0(x, epsilon=eps):
    """Berechnet die Funktionswerte von psi_0 für ein gegebenen Vektor x."""
    psi_0 = (1 / (2*np.pi)**(1/4)) * exp(-x**2 / 2) * exp(1j * x * px / epsilon)
    return psi_0


# Grundzustand der Eigenfunktionen der Elektronen
def calc_phi_0(x, y):
    """Berechnet die Funktionswerte von phi_0 für zwei gegebene Vektoren x und y."""
    Nx, Ny = len(x), len(y)
    phi_0 = np.empty((Ny, Nx))
    w = frequency(x)
    for j in range(Ny):
        for i in range(Nx):
            phi_0[j, i] = (w[i] / np.pi)**(1/4) * exp(-w[i] / 2 * y[j]**2)

    return phi_0


# Eigenfunktion phi_2
def calc_phi_2(x, y):
    """Berechnet die Funktionswerte von phi_2 für zwei gegebene Vektoren x und y."""
    Nx, Ny = len(x), len(y)
    phi_2 = np.empty((Ny, Nx))
    w = frequency(x)
    for j in range(Ny):
        for i in range(Nx):
            phi_2[j, i] = (sqrt(w[i]) / (8 * sqrt(np.pi)))**(1/2) * (4 * w[i] * y[j]**2 - 2) * exp(-w[i] / 2 * y[j]**2)

    return phi_2


# Wellenfunktion im 2D-Gesamtsystem
def calc_Psi(x, y, psi, phi_0, useU, epsilon=eps):
    """Berechnet die Funktionswerte der zweidimensionalen Wellenfunktion aus den gegebenen
    Funktionswerten von psi und phi."""
    Ny, Nx = np.shape(phi_0)
    Psi = np.empty((Ny, Nx), dtype=complex)

    if useU is True:
        phi_2 = calc_phi_2(x, y)
        w = frequency(x)
        w_derived = frequency_derived(x)
        k = 2 * np.pi * fftfreq(Nx, 2 * np.abs(x[0]) / Nx)
        psi_derived = ifft(1j * k * fft(psi))

        for j in range(Ny):
            for i in range(Nx):
                Psi[j, i] = psi[i] * phi_0[j, i] \
                            - epsilon**2 * w_derived[i]/(2**(5/2) * w[i]**2) * phi_2[j, i] * psi_derived[i]

    else:
        for j in range(Ny):
            for i in range(Nx):
                Psi[j, i] = psi[i] * phi_0[j, i]

    return Psi


# Potential
def potential(x, y):
    """Berechnet das Potential aus den Vektoren x und y."""
    Nx, Ny = len(x), len(y)
    V = np.empty((Ny, Nx))
    w = frequency(x)
    for j in range(Ny):
        for i in range(Nx):
            V[j, i] = -0.5 * x[i]**2 + 0.5 * w[i]**2 * y[j]**2

    return V


# Potential oben abschneiden
def potential_cut(V, c):
    """Schneidet alle Werte des gegebenen Potentials oberhalb von zwei ab."""
    Vy, Vx = V.shape
    Vcut = np.copy(V)
    for j in range(Vy):
        for i in range(Vx):
            if c < Vcut[j, i]:
                Vcut[j, i] = None

    return Vcut


# Konturplot des Potentials
def plot_potential(x, y, V):
    """Erstellt ein Konturplot eines gegebenen Potentials."""
    Vcut = potential_cut(V, 4)

    plt.figure()
    plot_norm = mpl.colors.SymLogNorm(linthresh=1.5, vmin=-20, vmax=4)
    plt.contourf(x, y, Vcut, 500, cmap=plt.cm.get_cmap('jet'), norm=plot_norm, extend='max')
    plt.colorbar(orientation='horizontal', format='% .0f', extend='max', extendfrac=0.02, shrink=0.85)
    plt.yticks(np.linspace(-4, 4, 5))
    plt.title('Konturplot von $V$')
    plt.xlabel('Longitudinale $x$')
    plt.ylabel('Transversale $y$')
    plt.show()


# HLS-Transformation
def complex_to_HSL(z):
    """Transformiert komplexe Funktionswerte in den HSL-Farbraum."""
    H = np.angle(z)
    L = (1 - 2 ** (-np.abs(z))) * 1
    S = 1

    c = np.vectorize(hls_to_rgb)(H, L, S)
    c = np.array(c)  # (3,n,m) zu (n,m,3) zu (m,n,3)
    c = c.swapaxes(0, 2)
    c = c.swapaxes(0, 1)

    return c


# HSL-Plot des Potentials
def plot_potentialHSL(V, xrange, yrange):
    """Erstellt ein HSL-Plot eines gegebenen Potentials."""
    plt.figure()
    img = complex_to_HSL(V)
    plt.imshow(img, aspect=0.68, extent=[-xrange, xrange, -yrange, yrange])
    plt.yticks(np.linspace(-4, 4, 5))
    plt.title('HSL-Plot von $V$')
    plt.xlabel('Longitudinale $x$')
    plt.ylabel('Transversale $y$')
    plt.show()


# Plot des Potentials
def plot_potential3D(x, y, V):
    """Erstellt ein 3D-Plot eines gegebenen Potentials."""
    Vcut = potential_cut(V, 10)

    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(X, Y, Vcut, rstride=32, cstride=30, alpha=0.5)
    zmin = Vcut[len(Vcut)//2, -1]
    ax.contour(X, Y, Vcut, zdir='z', offset=zmin, alpha=0.6)
    plt.yticks(np.linspace(-4, 4, 5))
    plt.title('3D-Plot von $V$')
    ax.set_xlabel('Longitudinale $x$')
    ax.set_ylabel('Transversale $y$')
    ax.set_zlabel('$V(x,y)$')
    plt.show()


# Konturplot der Wellenfunktion im Potential
def plot_potential_wave(x, y, V, allPsi):
    """Erstellt einzelne Konturplots einer Wellenfunktion zu verschiedenen Zeitpunkten in einem gegebenen Potential."""
    plt.figure(figsize=(8, 8))
    lvls = len(allPsi)

    if lvls == 1:
        Psi = np.abs(allPsi[0][1]) ** 2
        plt.contour(x, y, V, levels=np.arange(0.2, 2, 0.3), cmap=plt.cm.get_cmap('YlOrRd'))
        plt.contour(x, y, Psi, cmap=plt.cm.get_cmap('binary'))
        plt.ylim(-3, 3)
        plt.title('$t = {}$'.format(allPsi[0][0]))
        plt.xlabel('Longitudinale $x$')
        plt.ylabel('Transversale $y$')
    else:
        z = lvls // 2 + (lvls % 2 > 0)
        for i in range(1, lvls+1):
            plt.subplot(z, 2, i)
            plt.contour(x, y, V, levels=np.arange(0.2, 2, 0.3), cmap=plt.cm.get_cmap('YlOrRd'))
            plot_data = np.abs(allPsi[i - 1][1]) ** 2
            plt.contour(x, y, plot_data, cmap=plt.cm.get_cmap('binary'))
            plt.ylim(-2, 2)
            plt.yticks(np.linspace(-2, 2, 5))
            plt.title('$t = {}$'.format(allPsi[i-1][0]))
            if i == lvls:
                plt.xlabel('Longitudinale $x$')
            if (lvls % 2 == 0) and (i == lvls-1):
                plt.xlabel('Longitudinale $x$')
            if i % 2 != 0:
                plt.ylabel('Transversale $y$')

    plt.tight_layout()
    plt.show()


# HSL-Plot der Wellenfunktion
def plot_HSL_wave(allPsi, xrange, yrange):
    """Erstellt einzelne HSL-Plots einer Wellenfunktionen zu gegebenen Zeitpunkten."""
    plt.figure(figsize=(8, 8))
    lvls = len(allPsi)

    if lvls == 1:
        img = complex_to_HSL(allPsi[0][1])
        plt.imshow(img, aspect='auto', extent=[-xrange, xrange, -yrange, yrange])
        plt.ylim(-3, 3)
        plt.title('$t = {}$'.format(allPsi[0][0]))
        plt.xlabel('Longitudinale $x$')
        plt.ylabel('Transversale $y$')
    else:
        z = lvls // 2 + (lvls % 2 > 0)
        for i in range(1, lvls+1):
            plt.subplot(z, 2, i)
            img = complex_to_HSL(allPsi[i-1][1])
            plt.imshow(img, aspect='auto', extent=[-xrange, xrange, -yrange, yrange])
            plt.ylim(-3, 3)
            plt.yticks(np.linspace(-2, 2, 3))
            plt.title('$t = {}$'.format(allPsi[i-1][0]))
            if i == lvls:
                plt.xlabel('Longitudinale $x$')
            if (lvls % 2 == 0) and (i == lvls-1):
                plt.xlabel('Longitudinale $x$')
            if i % 2 != 0:
                plt.ylabel('Transversale $y$')

    plt.tight_layout()
    plt.show()


# Born-Huang-Potential
def potential_VBH(x):
    """Berechnet die Funktionswerte des Born-Huang-Potentials."""
    result = x**6 / (2 * (1 + x**4)**2)
    return result


# M-Term
def correction_term(x, epsilon):
    """Berechnet die Funktionswerte des Korrekturterms."""
    result = - epsilon**2 * x**6 / (4 * sqrt(2) * (1 + x**4)**(5/2))
    return result


# Plot Born-Huang-Potential
def plot_potential_VBH(x, VBH):
    """Plottet des Born-Huang-Potentials."""
    plt.figure()
    plt.plot(x, VBH)
    plt.title(r"Born-Huang-Potential $V_\mathrm{BH}$")
    plt.xlabel("$x$")
    plt.ylabel(r"$V_\mathrm{BH}(x)$")
    plt.show()


# Plot Born-Huang-Potential
def plot_correction_term(x, M):
    """Plottet den Korrekturterm."""
    plt.figure()
    plt.plot(x, M)
    plt.title(r"$\mathfrak{v}$-Term")
    plt.xlabel("$x$")
    plt.ylabel(r"$\mathfrak{v}(x)$")
    plt.show()


# adiabatische Zeitentwicklung nach Born-Oppenheimer (1d)
def time_propagation_BO(x, psi, t, useVBH=True, useM=True, epsilon=eps, t_delta=h):
    """Berechnet die adiabatische Zeitentwicklung mittels dem Born-Oppenheimer-Hamiltonoperator."""
    if t == 0:
        return psi

    E_0 = (sqrt(2 + 2 * x ** 4) - x ** 2) / 2
    if useVBH is True:
        VBH = potential_VBH(x)
    else:
        VBH = 0

    if abs(round(t / t_delta) - (t / t_delta)) <= 10 ** (-10):
        splitting_steps = int(round(t / t_delta))
    else:
        splitting_steps = int(np.ceil(t / t_delta))

    k = 2 * np.pi * fftfreq(len(x), 2*np.abs(x[0])/len(x))
    eA = exp(-1j/h_red * (E_0 + h_red**2/(2*m) * epsilon**2 * VBH) * t/(epsilon*2*splitting_steps))
    if useM is True:
        eB = exp(-1j/h_red * epsilon**2 * h_red**2/(2*m) * k**2 * t/(epsilon*2*splitting_steps))
        M = correction_term(x, epsilon)

        for i in range(splitting_steps):
            psi = eB * fft(eA * psi)
            pmp = ifft(1j * k * psi)
            if str(np.isnan(np.sum(pmp))) == 'True':
                raise ValueError('Bitte kleinere Schrittweite wählen! (Aktuell: h =' + str(h) + ')')
            pmp = 1j * k * fft(1j/h_red * epsilon * M * t / splitting_steps * pmp)
            psi += pmp
            psi = eA * (ifft(eB * psi))
    else:
        eB = exp(-1j/h_red * epsilon**2 * h_red**2/(2*m) * k**2 * t/(epsilon*splitting_steps))
        for i in range(splitting_steps):
            psi = eA * (ifft(eB * fft(eA * psi)))

    return psi


# Zeitentwicklung (2d)
def time_propagation_2d(x, y, Psi, V, t, epsilon=eps, t_delta=h):
    """Berechnet die Zeitentwicklung mittels dem zweidimensionalen Hamiltonoperator."""
    if t == 0:
        return Psi

    Nx, Ny = len(x), len(y)
    kx_tmp = 2 * np.pi * fftfreq(len(x), 2*np.abs(x[0])/len(x))
    ky_tmp = 2 * np.pi * fftfreq(len(y), 2*np.abs(y[0])/len(y))
    kx = np.empty((Ny, Nx))
    ky = np.empty((Ny, Nx))

    for j in range(Ny):
        kx[j, :] = kx_tmp
    for i in range(Nx):
        ky[:, i] = ky_tmp

    if abs(round(t / t_delta) - (t / t_delta)) <= 10 ** (-10):
        splitting_steps = int(round(t / t_delta))
    else:
        splitting_steps = int(np.ceil(t / t_delta))

    eA = exp(-1j/h_red * (epsilon**2 * h_red**2/(2*m) * kx**2 + h_red**2/(2*m) * ky**2) * t/(epsilon*splitting_steps))
    eB = exp(-1j/h_red * V * t/(epsilon*2*splitting_steps))
    for i in range(splitting_steps):
        Psi = eB * (ifft2(eA * fft2(eB * Psi)))

    return Psi


# Vektor aus Zeitentwicklungen
def multiple_propagation_times(x, y, times, *potV, effektiv=True,
                               useVBH=True, useM=True, useU=None, epsilon=eps, t_delta=h):
    """Berechnet die Zeitentwicklungen zu mehreren Zeitpunkten."""
    psi_0 = calc_psi_0(x, epsilon=epsilon)
    phi_0 = calc_phi_0(x, y)
    if effektiv is False:
        if len(potV) > 1:
            print('Warnung: Zu viele Argumente! Überflüssige werden ignoriert.')
        elif len(potV) == 0 or potV[0] is None:
            raise IOError('Kein Potential angegeben!')
        V = potV[0]
        if useU is None:
            useU = False
        Psi = calc_Psi(x, y, psi_0, phi_0, useU, epsilon=epsilon)
        typ = "2D"
    else:
        if len(potV) > 0 and potV[0] is not None:
            print('Warnung: Zu viele Argumente! Überflüssiges Potential wird ignoriert.')
        if useU is None:
            if useM is True:
                useU = True
            else:
                useU = False
        psi = psi_0
        typ = "1D"

    allPsi = []
    print('\t[' + typ + ']   Todo: ' + '[' + 100 * '-' + ']')
    sys.stdout.write('\tFortschritt: [')
    sys.stdout.flush()
    printedMark = 0
    start = time.time()
    for i in range(len(times)):
        if effektiv is False:
            if i >= 1:
                Psi = time_propagation_2d(x, y, Psi, V, times[i]-times[i-1], epsilon=epsilon, t_delta=t_delta)
            else:
                Psi = time_propagation_2d(x, y, Psi, V, times[i], epsilon=epsilon, t_delta=t_delta)
        else:
            if i >= 1:
                psi = time_propagation_BO(x, psi, times[i]-times[i-1], useVBH, useM, epsilon=epsilon, t_delta=t_delta)
            else:
                psi = time_propagation_BO(x, psi, times[i], useVBH, useM, epsilon=epsilon, t_delta=t_delta)
            Psi = calc_Psi(x, y, psi, phi_0, useU, epsilon=epsilon)
        allPsi.append([times[i], Psi])
        if len(times)-1 == 0:
            currentMark = 100
        else:
            currentMark = 100 * i // (len(times)-1)
        if printedMark < currentMark:
            sys.stdout.write('*' * (currentMark - printedMark))
            sys.stdout.flush()
            printedMark = currentMark
    sys.stdout.write(']\n')
    end = time.time()
    total = [int((end - start) % 60), (end - start) // 60]                              # [s, min, (h)]
    if total[1] >= 60:
        total.append(total[1] // 60)
        total[1] = int(total[1] % 60)
        print('\tBenötigte Zeit: {:02.0f}:{:02d}:{:02d} Stunden\n'.format(total[2], total[1], total[0]))
    else:
        print('\tBenötigte Zeit: {:02.0f}:{:02d} Minuten\n'.format(total[1], total[0]))

    return allPsi


# Vergleiche Zeitentwicklungen bei unterschiedlichen Gittergrößen
def compare_grid(xrange, yrange, times, *nx, ny=None, potV=None,
                 effektiv=True, useVBH=True, useM=True, useU=False, handling="new"):
    """Berechnet die normierte Differenz der Wellenfunktionen zu gegebenen Gittergrößen."""
    if effektiv is False and handling[0] == "load":
        with open(handling[1], 'rb') as input_file:
            print("\t2D-Daten werden geladen...")
            nx, ny, allPsi_nxy = pickle.load(input_file)
    else:
        x_nx, y_ny, allPsi_nxy = [], [], []
        V = None

        if ny is None:
            empty = True
            ny = []
        else:
            empty = False

        for i in range(len(nx)):
            x_nx.append(np.arange(-xrange, xrange, 2 * xrange / (2 ** nx[i])))          # Intervall x-Richtung
            if empty:
                ny.append(nx[i])
                y_ny.append(np.arange(-yrange, yrange, 2 * yrange / (2 ** nx[i])))      # Intervall y-Richtung
            else:
                if i < len(ny):
                    y_ny.append(np.arange(-yrange, yrange, 2 * yrange / (2 ** ny[i])))  # Intervall y-Richtung
                else:
                    print("Warnung: Weniger y-Gittergrößen als x-Gittergrößen angegeben! Verwende ny = nx.")
                    ny.append(nx[i])
                    y_ny.append(np.arange(-yrange, yrange, 2 * yrange / (2 ** ny[i])))  # Intervall y-Richtung
            if effektiv is False:
                if potV is None:
                    raise IOError('Keine Potentialfunktion angegeben!')
                else:
                    V = potV(x_nx[i], y_ny[i])
            allPsi_nxy.append(multiple_propagation_times(x_nx[i], y_ny[i], times, V, effektiv=effektiv,
                                                         useVBH=useVBH, useM=useM, useU=useU))
    if effektiv is False and handling[0] == "save":
        with open(handling[1], 'wb') as input_file:
            print("\t2D-Daten werden gespeichert in " + handling[1] + "\n")
            pickle.dump([nx, ny, allPsi_nxy], input_file)

    normDiff = np.empty((len(nx)-1, len(times)))
    for i in range(len(nx)-1):
        for t in range(len(times)):
            diff_x = np.abs(nx[i+1] - nx[i])
            diff_y = np.abs(ny[i + 1] - ny[i])
            if nx[i] <= nx[i+1] and ny[i] <= ny[i+1]:
                normFact = sqrt((4 * xrange * yrange) / (nx[i] * ny[i]))
                normDiff[i][t] = normFact * np.linalg.norm(
                    allPsi_nxy[i][t][1] - (allPsi_nxy[i + 1][t][1])[::2**diff_y, ::2**diff_x])
            elif nx[i] <= nx[i+1] and ny[i] > ny[i+1]:
                normFact = sqrt((4 * xrange * yrange) / (nx[i] * ny[i + 1]))
                normDiff[i][t] = normFact * np.linalg.norm(
                    (allPsi_nxy[i][t][1])[::2 ** diff_y, :] - (allPsi_nxy[i + 1][t][1])[:, ::2 ** diff_x])
            elif nx[i] > nx[i + 1] and ny[i] <= ny[i + 1]:
                normFact = sqrt((4 * xrange * yrange) / (nx[i + 1] * ny[i]))
                normDiff[i][t] = normFact * np.linalg.norm(
                    (allPsi_nxy[i][t][1])[:, ::2 ** diff_x] - (allPsi_nxy[i + 1][t][1])[::2 ** diff_y, :])
            else:
                normFact = sqrt((4 * xrange * yrange) / (nx[i + 1] * ny[i + 1]))
                normDiff[i][t] = normFact * np.linalg.norm(
                    (allPsi_nxy[i][t][1])[::2**diff_y, ::2**diff_x]) - allPsi_nxy[i + 1][t][1]

    return normDiff


# Plot des Gittergrößen-Vergleichs
def plot_grid(normDiff, normDiff_eff, times, *nx, ny=None):
    """Plottet bis zu vier Ergebnisse des Vergleichs von den (bis zu fünf) Gittergrößen."""
    if len(nx) > 5 or len(nx) <= 1:
        print("Zu wenig/viele Gittergrößen angegeben! (Nötig: 2 bis 5)")
        print("Abbruch.")
        return
    else:
        if ny is not None and len(nx) != len(ny):
            print("Zu wenig/viele Gittergrößen angegeben! (Nötig: 2 bis 5)")
            print("Abbruch.")
            return

    plt.figure()
    if ny is None:
        lbl = r'$\mathrm{{ges:}}$ ' + r'$ m_x = {0}, n_x = {1}$'
        lbl_eff = r'$\mathrm{{eff:}}$ ' + r'$ m_x = {0}, n_x = {1}$'
        plt.ylabel(r'$\Vert \mathit{\Psi}^\varepsilon_{m_x} - \mathit{\Psi}^\varepsilon_{n_x} \Vert_{L^2}$')
        nxy = nx
    else:
        lbl = r'$\mathrm{{ges:}}$ ' + r'$ m_y = {0}, n_y = {1}$'
        lbl_eff = r'$\mathrm{{eff:}}$ ' + r'$ m_y = {0}, n_y = {1}$'
        plt.ylabel(r'$\Vert \mathit{\Psi}^\varepsilon_{m_y} - \mathit{\Psi}^\varepsilon_{n_y} \Vert_{L^2}$')
        nxy = ny

    plt.plot(times, normDiff_eff[0], c='orange', marker='v', mec='None', ls='', label=lbl_eff.format(nxy[0], nxy[1]))
    plt.plot(times, normDiff[0], color='orange', marker='+', ls='', label=lbl.format(nxy[0], nxy[1]))
    if len(nxy) >= 3:
        plt.plot(times, normDiff_eff[1], c='firebrick', marker='v', mec='None', ls='',
                 label=lbl_eff.format(nxy[1], nxy[2]))
        plt.plot(times, normDiff[1], color='firebrick', marker='+', ls='', label=lbl.format(nxy[1], nxy[2]))
    if len(nxy) >= 4:
        plt.plot(times, normDiff_eff[2], 'bv', markeredgecolor='None', ls='', label=lbl_eff.format(nxy[2], nxy[3]))
        plt.plot(times, normDiff[2], 'b+', label=lbl.format(nxy[2], nxy[3]))
    if len(nxy) >= 5:
        plt.plot(times, normDiff_eff[3], 'gv', markeredgecolor='None', ls='', label=lbl_eff.format(nxy[3], nxy[4]))
        plt.plot(times, normDiff[3], 'g+', label=lbl.format(nxy[3], nxy[4]))

    plt.yscale('symlog', linthreshy=10 ** (-9))
    plt.legend(loc='best', numpoints=1, ncol=2, columnspacing=1, handletextpad=0)
    plt.xlabel('Zeit $t$')
    plt.show()


# Vergleiche Zeitentwicklungen bei unterschiedlicher Schrittgröße beim Strang-Splitting
def compare_splitting_steps(x, y, times, *hs, potV=None, effektiv=True,
                            useVBH=True, useM=True, useU=False, handling="new"):
    """Berechnet die normierte Differenz der Wellenfunktionen zu gegebenen Schrittweiten des Strang-Splittings"""
    if effektiv is False and handling[0] == "load":
        with open(handling[1], 'rb') as input_file:
            print("\t2D-Daten werden geladen...")
            allPsi_n = pickle.load(input_file)
    else:
        allPsi_n = []
        for i in range(len(hs)):
            allPsi_n.append(multiple_propagation_times(
                x, y, times, potV, effektiv=effektiv, useVBH=useVBH, useM=useM, useU=useU, t_delta=hs[i]))
    if effektiv is False and handling[0] == "save":
        with open(handling[1], 'wb') as input_file:
            print("\t2D-Daten werden gespeichert in " + handling[1] + "\n")
            pickle.dump(allPsi_n, input_file)

    normDiff = np.empty((len(hs)-1, len(times)))
    normFact = sqrt((4 * np.abs(x[0]) * np.abs(y[0])) / (len(x) * len(y)))
    for i in range(len(hs)-1):
        for t in range(len(times)):
            normDiff[i][t] = normFact * np.linalg.norm(allPsi_n[i + 1][t][1] - allPsi_n[i][t][1])

    return normDiff


# Plot des Strang-Splitting-Vergleichs
def plot_splitting(normDiff, normDiff_eff, times, *hs):
    """Plottet bis zu vier Ergebnisse des Vergleichs von den (bis zu fünf) Schrittweiten im Strang-Splitting."""
    if len(hs) > 5 or len(hs) <= 1:
        print("Zu wenig/viele Schrittgrößen angegeben! (Nötig: 2 bis 5)")
        print("Abbruch.")
        return

    hs_frac = [Fraction(hs[i]).limit_denominator() for i in range(len(hs))]
    hs_lbl = [r'\frac{{{0}}}{{{1}}}'.format(hs_frac[i].numerator, hs_frac[i].denominator) for i in range(len(hs_frac))]
    lbl = r'$\mathrm{{ges:}}$ ' + r'$ h_1 = {0}, h_2 = {1}$'
    lbl_eff = r'$\mathrm{{eff:}}$ ' + r'$ h_1 = {0}, h_2 = {1}$'

    plt.figure()
    plt.plot(times, normDiff_eff[0], c='orange', marker='v', mec='None', ls='',
             label=lbl_eff.format(hs_lbl[0], hs_lbl[1]))
    plt.plot(times, normDiff[0], color='orange', marker='+', ls='', label=lbl.format(hs_lbl[0], hs_lbl[1]))
    if len(hs) >= 3:
        plt.plot(times, normDiff_eff[1], c='firebrick', marker='v', mec='None', ls='',
                 label=lbl_eff.format(hs_lbl[1], hs_lbl[2]))
        plt.plot(times, normDiff[1], color='firebrick', marker='+', ls='', label=lbl.format(hs_lbl[1], hs_lbl[2]))
    if len(hs) >= 4:
        plt.plot(times, normDiff_eff[2], 'bv', markeredgecolor='None', ls='',
                 label=lbl_eff.format(hs_lbl[2], hs_lbl[3]))
        plt.plot(times, normDiff[2], 'b+', label=lbl.format(hs_lbl[2], hs_lbl[3]))
    if len(hs) == 5:
        plt.plot(times, normDiff_eff[3], 'gv', markeredgecolor='None', ls='',
                 label=lbl_eff.format(hs_lbl[3], hs_lbl[4]))
        plt.plot(times, normDiff[3], 'g+', label=lbl.format(hs_lbl[3], hs_lbl[4]))

    plt.yscale('symlog', linthreshy=10 ** (-9))
    plt.legend(loc='best', numpoints=1, ncol=2, columnspacing=1, handletextpad=0)
    plt.xlabel('Zeit $t$')
    plt.ylabel(r'$\Vert \mathit{\Psi}^\varepsilon_{h_1} - \mathit{\Psi}^\varepsilon_{h_2} \Vert_{L^2}$')
    plt.show()


# Vektor aus Zeitentwicklung bei verschiedenen epsilon zu einem festem Zeitpunkt
def multiple_propagation_eps(x, y, t, epsilons, *potV, effektiv=True, useVBH=True, useM=True, useU=None):
    """Berechnet die Zeitentwicklung bezüglich verschiedener epsilon zu einem festen Zeitpunkten."""
    if effektiv is False:
        if len(potV) > 1:
            print('Warnung: Zu viele Argumente! Überflüssige werden ignoriert.')
        elif len(potV) == 0 or potV[0] is None:
            raise IOError('Kein Potential angegeben!')
        V = potV[0]
        if useU is None:
            useU = False
        typ = "2D"
    else:
        if len(potV) > 0 and potV[0] is not None:
            print('Warnung: Zu viele Argumente! Überflüssiges Potential wird ignoriert.')
        if useU is None:
            if useM is True:
                useU = True
            else:
                useU = False
        typ = "1D"

    allPsi = []
    phi_0 = calc_phi_0(x, y)
    print('\t[' + typ + ']   Todo: ' + '[' + 100 * '-' + ']')
    sys.stdout.write('\tFortschritt: [')
    sys.stdout.flush()
    printedMark = 0
    start = time.time()
    for i in range(len(epsilons)):
        psi_0 = calc_psi_0(x, epsilons[i])
        if effektiv is False:
            Psi_0 = calc_Psi(x, y, psi_0, phi_0, useU, epsilon=epsilons[i])
            Psi = time_propagation_2d(x, y, Psi_0, V, t, epsilon=epsilons[i])
        else:
            psi = time_propagation_BO(x, psi_0, t, useVBH, useM, epsilon=epsilons[i])
            Psi = calc_Psi(x, y, psi, phi_0, useU, epsilon=epsilons[i])
        allPsi.append([epsilons[i], Psi])
        if len(epsilons)-1 == 0:
            currentMark = 100
        else:
            currentMark = 100 * i // (len(epsilons)-1)
        if printedMark < currentMark:
            sys.stdout.write('*' * (currentMark - printedMark))
            sys.stdout.flush()
            printedMark = currentMark
    sys.stdout.write(']\n')
    end = time.time()
    total = [int((end - start) % 60), (end - start) // 60]                  # [s, min, (h)]
    if total[1] >= 60:
        total.append(total[1] // 60)
        total[1] = int(total[1] % 60)
        print('\tBenötigte Zeit: {:02.0f}:{:02d}:{:02d} Stunden\n'.format(total[2], total[1], total[0]))
    else:
        print('\tBenötigte Zeit: {:02.0f}:{:02d} Minuten\n'.format(total[1], total[0]))

    return allPsi


# Vergleiche Zeitentwicklungen bei unterschiedlichen Epsilons
def compare_error_eps(x, y, t, epsilons, V, handling="new"):
    """Berechnet die Fehler der Born-Oppenheimer-Approximation mit und ohne Korrekturterm zu gegebenen Epsilons."""
    if handling[0] == "load":
        with open(handling[1], 'rb') as input_file:
            print("\t2D-Daten werden geladen...")
            allPsi, allPsiU = pickle.load(input_file)
    else:
        allPsi = multiple_propagation_eps(x, y, t, epsilons, V, effektiv=False)
        allPsiU = multiple_propagation_eps(x, y, t, epsilons, V, effektiv=False, useU=True)
    if handling[0] == "save":
        with open(handling[1], 'wb') as input_file:
            print("\t2D-Daten werden gespeichert in " + handling[1] + "\n")
            pickle.dump([allPsi, allPsiU], input_file)

    allPsi_eff = multiple_propagation_eps(x, y, t, epsilons, useVBH=False, useM=False, useU=False)
    allPsi_effV = multiple_propagation_eps(x, y, t, epsilons, useM=False, useU=False)
    allPsi_effVM = multiple_propagation_eps(x, y, t, epsilons, useU=False)
    allPsi_effVMU = multiple_propagation_eps(x, y, t, epsilons)

    normDiff_eff = np.empty(len(epsilons))
    normDiff_effV = np.empty(len(epsilons))
    normDiff_effVM = np.empty(len(epsilons))
    normDiff_effVMU = np.empty(len(epsilons))
    normFact = sqrt((4 * np.abs(x[0]) * np.abs(y[0])) / (len(x) * len(y)))
    for i in range(len(epsilons)):
        normDiff_eff[i] = normFact * np.linalg.norm(allPsi[i][1] - allPsi_eff[i][1])
        normDiff_effV[i] = normFact * np.linalg.norm(allPsi[i][1] - allPsi_effV[i][1])
        normDiff_effVM[i] = normFact * np.linalg.norm(allPsi[i][1] - allPsi_effVM[i][1])
        normDiff_effVMU[i] = normFact * np.linalg.norm(allPsiU[i][1] - allPsi_effVMU[i][1])

    return normDiff_eff, normDiff_effV, normDiff_effVM, normDiff_effVMU


# Plot des Epsilon-Vergleichs
def plot_error_eps(normDiff_eff, normDiff_effV, normDiff_effVM, normDiff_effVMU, t, epsilons, show_bound=True):
    """Plottet die Fehler der Born-Oppenheimer-Approximation mit und ohne Korrekturterm zu gegebenen Epsilons."""
    plt.figure()
    if show_bound is True:
        c1, c2 = [], []
        for i in range(len(epsilons)):
            c1.append(normDiff_effV[i] / (epsilons[i] * (1 + abs(t))))
            c2.append(normDiff_effVMU[i] / (epsilons[i]**2 * (1+abs(t))))
        c1 = np.amax(c1)
        c2 = np.amax(c2)
        plt.plot(np.log2(epsilons), np.log2(epsilons) + np.log2(c1 * (1 + abs(t))), c='gray', ls='dotted')
        plt.plot(np.log2(epsilons), 2 * np.log2(epsilons) + np.log2(c2 * (1 + abs(t))), c='gray', ls='dotted')

    plt.plot(np.log2(epsilons), np.log2(normDiff_eff), c='firebrick', marker='v', mec='None',
             ls='--', label=r'ohne $V_{\mathrm{BH}}$ und $\mathcal{M}$')
    plt.plot(np.log2(epsilons), np.log2(normDiff_effV), 'bv--', mec='None', label=r'ohne $\mathcal{M}$')
    plt.plot(np.log2(epsilons), np.log2(normDiff_effVM), c='orange', marker='v', mec='None',
             ls='--', label=r'mit $\mathcal{M}$, ohne $U_{(1)}^{\varepsilon\, *}$')
    plt.plot(np.log2(epsilons), np.log2(normDiff_effVMU), 'gv--', mec='None',
             label=r'mit $\mathcal{M}$ und $U_{(1)}^{\varepsilon\, *}$')
    plt.legend(loc='best')
    plt.title('$t = {}$'.format(t))
    plt.xlabel(r'$\log_2(\varepsilon)$')
    plt.ylabel(r'$\log_2(\Vert \mathit{\Psi}^\varepsilon - \mathit{\Psi}^\varepsilon_{\mathrm{eff}} \Vert_{L^2})$')
    plt.show()


# Vergleiche die Entwicklung des Fehlers über die Zeit
def compare_error_time(x, y, times, V, handling="new"):
    """Berechnet den Fehler der Born-Oppenheimer-Approximation mit und ohne Korrekturterm zu gegebenen Zeiten."""
    if handling[0] == "load":
        with open(handling[1], 'rb') as input_file:
            print("\t2D-Daten werden geladen...")
            allPsi, allPsiU = pickle.load(input_file)
    else:
        allPsi = multiple_propagation_times(x, y, times, V, effektiv=False)
        allPsiU = multiple_propagation_times(x, y, times, V, effektiv=False, useU=True)
    if handling[0] == "save":
        with open(handling[1], 'wb') as input_file:
            print("\t2D-Daten werden gespeichert in " + handling[1] + "\n")
            pickle.dump([allPsi, allPsiU], input_file)

    allPsi_eff = multiple_propagation_times(x, y, times, useVBH=False, useM=False, useU=False)
    allPsi_effV = multiple_propagation_times(x, y, times, useM=False, useU=False)
    allPsi_effVM = multiple_propagation_times(x, y, times, useU=False)
    allPsi_effVMU = multiple_propagation_times(x, y, times)

    normDiff_eff = np.empty(len(times))
    normDiff_effV = np.empty(len(times))
    normDiff_effVM = np.empty(len(times))
    normDiff_effVMU = np.empty(len(times))
    normFact = sqrt((4 * np.abs(x[0]) * np.abs(y[0])) / (len(x) * len(y)))
    for i in range(len(times)):
        normDiff_eff[i] = normFact * np.linalg.norm(allPsi[i][1] - allPsi_eff[i][1])
        normDiff_effV[i] = normFact * np.linalg.norm(allPsi[i][1] - allPsi_effV[i][1])
        normDiff_effVM[i] = normFact * np.linalg.norm(allPsi[i][1] - allPsi_effVM[i][1])
        normDiff_effVMU[i] = normFact * np.linalg.norm(allPsiU[i][1] - allPsi_effVMU[i][1])

    return normDiff_eff, normDiff_effV, normDiff_effVM, normDiff_effVMU


# Plot des Fehlers über die Zeit
def plot_error_time(times, normDiff_eff, normDiff_effV, normDiff_effVM, normDiff_effVMU):
    """Plottet die Fehler der Born-Oppenheimer-Approximation mit und ohne Korrekturterm zu gegebenen Zeiten."""
    plt.figure()
    plt.plot(times, normDiff_eff, c='firebrick', marker='v', mec='None',
             ls='--', label=r'ohne $V_{\mathrm{BH}}$ und $\mathcal{M}$')
    plt.plot(times, normDiff_effV, 'bv--', mec='None', label=r'ohne $\mathcal{M}$')
    plt.plot(times, normDiff_effVM, c='orange', marker='v', mec='None', ls='--',
             label=r'mit $\mathcal{M}$, ohne $U_{(1)}^{\varepsilon\, *}$')
    plt.plot(times, normDiff_effVMU, 'gv--', mec='None', label=r'mit $\mathcal{M}$ und $U_{(1)}^{\varepsilon\, *}$')
    plt.legend(loc='best')
    plt.xlabel('Zeit $t$')
    plt.ylabel(r'$\Vert \mathit{\Psi}^\varepsilon - \mathit{\Psi}^\varepsilon_{\mathrm{eff}} \Vert_{L^2}$')
    plt.show()


# Vergleiche die Entwicklung des Fehlers der Approximation zweiter Ordnung über die Zeit
def compare_error_time_VMU(x, y, times, epsilons, V, handling="new"):
    """Berechnet den Fehler der Born-Oppenheimer-Approximation zweiter Ordnung zu gegebenen Zeiten."""
    if handling[0] == "load":
        with open(handling[1], 'rb') as input_file:
            print("\t2D-Daten werden geladen...")
            allPsiU = pickle.load(input_file)
    else:
        allPsiU = []
        for i in range(len(epsilons)):
            PsiU = multiple_propagation_times(x, y, times, V, effektiv=False, useU=True, epsilon=epsilons[i])
            allPsiU.append([epsilons[i], PsiU])
    if handling[0] == "save":
        with open(handling[1], 'wb') as input_file:
            print("\t2D-Daten werden gespeichert in " + handling[1] + "\n")
            pickle.dump(allPsiU, input_file)

    allPsi_effVMU = []
    for i in range(len(epsilons)):
        Psi_effVMU = multiple_propagation_times(x, y, times, epsilon=epsilons[i])
        allPsi_effVMU.append([epsilons[i], Psi_effVMU])

    normDiff_effVMU_eps = []
    normDiff_effVMU = np.empty(len(times))
    normFact = sqrt((4 * np.abs(x[0]) * np.abs(y[0])) / (len(x) * len(y)))

    for j in range(len(epsilons)):
        for i in range(len(times)):
            normDiff_effVMU[i] = normFact * np.linalg.norm(allPsiU[j][1][i][1] - allPsi_effVMU[j][1][i][1])
        normDiff_effVMU_eps.append([epsilons[j], np.copy(normDiff_effVMU)])

    return normDiff_effVMU_eps


# Plot des Fehlers der zweiten Ordnung über die Zeit
def plot_error_time_VMU(times, normDiff_effVMU_eps, interpolate=False):
    """Plottet bis zu drei Fehlerkurven der Born-Oppenheimer-Approximation zweiter Ordnung."""
    if len(normDiff_effVMU_eps) > 3 or len(normDiff_effVMU_eps) < 1:
        print("Zu wenig/viele Argumente angegeben! (Zulässig: 1 bis 4)")
        print("Abbruch.")
        return

    epsilons = [normDiff_effVMU_eps[i][0] for i in range(len(normDiff_effVMU_eps))]
    eps_frac = [Fraction(epsilons[i]).limit_denominator() for i in range(len(epsilons))]

    plt.figure()
    plt.plot(times, normDiff_effVMU_eps[0][1], 'gs--', mec='None',
             label=r'$\varepsilon=\frac{{{0}}}{{{1}}}$'.format(eps_frac[0].numerator, eps_frac[0].denominator))
    if interpolate is True:
        c = np.poly1d(np.polyfit(times[1:], normDiff_effVMU_eps[0][1][1:], 1))
        print("Interpolationsgerade für eps = " + str(eps_frac[0].numerator) + "/" + str(
            eps_frac[0].denominator) + ":")
        print("\t(" + str(eps_frac[0].numerator) + "/" + str(eps_frac[0].denominator) + ")^2 * (" + str(
            normDiff_effVMU_eps[0][0] ** (-2) * c[1]) + " * t + " + str(normDiff_effVMU_eps[0][0] ** (-2) * c[0]) + ")")
        plt.plot(times[1:], c(times[1:]), 'darkorange')
    if len(normDiff_effVMU_eps) >= 2:
        plt.plot(times, normDiff_effVMU_eps[1][1], 'gd--', mec='None',
                 label=r'$\varepsilon=\frac{{{0}}}{{{1}}}$'.format(eps_frac[1].numerator, eps_frac[1].denominator))
        if interpolate is True:
            c = np.poly1d(np.polyfit(times[1:], normDiff_effVMU_eps[1][1][1:], 1))
            print("Interpolationsgerade für eps = " + str(eps_frac[1].numerator) + "/" + str(
                eps_frac[1].denominator) + ":")
            print("\t(" + str(eps_frac[1].numerator) + "/" + str(eps_frac[1].denominator) + ")^2 * (" + str(
                normDiff_effVMU_eps[1][0] ** (-2) * c[1]) + " * t + " + str(
                normDiff_effVMU_eps[1][0] ** (-2) * c[0]) + ")")
            plt.plot(times[1:], c(times[1:]), 'darkorange')
    if len(normDiff_effVMU_eps) == 3:
        plt.plot(times, normDiff_effVMU_eps[2][1], 'gv--', mec='None',
                 label=r'$\varepsilon=\frac{{{0}}}{{{1}}}$'.format(eps_frac[2].numerator, eps_frac[2].denominator))
        if interpolate is True:
            c = np.poly1d(np.polyfit(times[1:], normDiff_effVMU_eps[2][1][1:], 1))
            print("Interpolationsgerade für eps = " + str(eps_frac[2].numerator) + "/" + str(
                eps_frac[2].denominator) + ":")
            print("\t(" + str(eps_frac[2].numerator) + "/" + str(eps_frac[2].denominator) + ")^2 * (" + str(
                normDiff_effVMU_eps[2][0] ** (-2) * c[1]) + " * t + " + str(
                normDiff_effVMU_eps[2][0] ** (-2) * c[0]) + ")")
            plt.plot(times[1:], c(times[1:]), 'darkorange')

    plt.legend(loc='best')
    plt.xlabel('Zeit $t$')
    plt.ylabel(r'$\Vert \mathit{\Psi}^\varepsilon - \mathit{\Psi}^\varepsilon_{\mathrm{BO}} \Vert_{L^2}$')
    plt.show()


# Abfrage zur Datenerhebung
def ask_data_handling():
    """Frägt ab, ob die nötigen Daten eingelesen oder neu berechnet, sowie anschließend gespeichert werden sollen."""
    retries = 3
    while True:
        ask = str(input("""
      Sollen die Daten neu berechnet oder eingelesen werden?
          [1] 2D-Daten neu berechnen
          [2] 2D-Daten einlesen
          -----------------------
          [Z] Zurück
      >> """))

        if ask == "1":
            while True:
                ask = str(input("""
        Sollen die 2D-Daten anschließend gespeichert werden [J/n]? ([Z] Zurück)
        >> """))
                if ask in ("Z", "z"):
                    break
                elif ask in ("n", "N"):
                    return "new"
                else:
                    while True:
                        ask = str(input("""
          Bitte Speicherpfad angeben: [./wave2d.pkl] ([Z] Zurück)
          >> """))
                        if ask in ("Z", "z"):
                            break
                        elif ask == '':
                            path = "./wave2d.pkl"
                        else:
                            path = ask
                        if os.path.exists(path) is False:
                            return ["save", path]
                        else:
                            while True:
                                ask = str(input("""
            Datei existiert bereits! Überschreiben [j/N]? ([Z] Zurück)
            >> """))
                                if ask in ("j", "J"):
                                    return ["save", path]
                                else:
                                    break
        elif ask == "2":
            while True:
                ask = str(input("""
        Bitte Quelldatei angeben: [./wave2d.pkl] ([Z] Zurück, [A] Ordnerinhalt anzeigen)
        Sicherheitswarnung: Nur vertrauenswürdige Dateien verwenden!
        >> """))
                if ask in ("Z", "z"):
                    break
                elif ask in ("A", "a"):
                    print(8 * " " + str(os.listdir("./")))
                    continue
                elif ask == '':
                    path = "./wave2d.pkl"
                else:
                    path = ask
                if os.path.exists(path) is False:
                    print(8 * " " + "Datei existiert nicht!")
                    continue
                return ["load", path]
        elif ask in ("Z", "z"):
            return
        else:
            print(6 * " " + "Ungültige Eingabe!")

        retries -= 1
        if retries == 0:
            return
