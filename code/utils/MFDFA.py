import numpy as np
import pandas as pd


class ComputeMFDFA:
    @staticmethod
    def mfdfa(signal, scales, q_list, order=1):
        """
        Calcule le MF-DFA pour une série temporelle.

        Paramètres:
            signal : tableau numpy, la série (par exemple, rendements) à analyser.
            scales : liste des échelles (tailles de segments) à utiliser.
            q_list : liste des ordres q pour lesquels calculer la fonction de fluctuation.
            order : ordre du polynôme pour le detrending (1 = linéaire par défaut).

        Retourne:
            Fq : matrice de taille (len(q_list), len(scales)) contenant F_q(s) pour chaque q et chaque échelle s.
        """
        N = len(signal)
        # Centrer et intégrer le signal (profil)
        signal = signal - np.mean(signal)
        Y = np.cumsum(signal)
        Fq = np.zeros((len(q_list), len(scales)))

        # Pour chaque échelle s
        for j, s in enumerate(scales):
            s = int(s)
            if s < 2:
                continue
            n_segments = N // s
            F_seg = []
            # Découpage non chevauchant depuis le début
            for v in range(n_segments):
                segment = Y[v * s:(v + 1) * s]
                idx = np.arange(s)
                coeffs = np.polyfit(idx, segment, order)
                fit = np.polyval(coeffs, idx)
                F_seg.append(np.mean((segment - fit) ** 2))

            # Découpage depuis la fin pour couvrir la totalité de la série
            for v in range(n_segments):
                segment = Y[N - (v + 1) * s:N - v * s]
                idx = np.arange(s)
                coeffs = np.polyfit(idx, segment, order)
                fit = np.polyval(coeffs, idx)
                F_seg.append(np.mean((segment - fit) ** 2))

            F_seg = np.array(F_seg)
            # Éviter les valeurs nulles
            F_seg[F_seg < 1e-10] = 1e-10

            # Calcul de F_q(s) pour chaque valeur de q
            for k, q in enumerate(q_list):
                if np.abs(q) < 1e-6:
                    # q = 0 : utilisation de la moyenne géométrique (limite q->0)
                    Fq[k, j] = np.exp(0.5 * np.mean(np.log(F_seg)))
                else:
                    Fq[k, j] = (np.mean(F_seg ** (q / 2))) ** (1 / q)
        return Fq

    @staticmethod
    def compute_alpha_falpha(q_list, h_q):
        """
        Calcule alpha et f(alpha) via la transformation de Legendre
        à partir de h(q).

        Paramètres:
            q_list : liste des ordres q.
            h_q    : tableau des exposants h(q) calculés.

        Retourne:
            alpha   : tableau des valeurs alpha.
            f_alpha : tableau des valeurs f(alpha).
        """
        dq = q_list[1] - q_list[0]
        dh_dq = np.gradient(h_q, dq)  # dérivée numérique de h(q)
        alpha = h_q + q_list * dh_dq  # alpha(q) = h(q) + q * h'(q)
        f_alpha = q_list * (alpha - h_q) + 1  # f(alpha) = q * [alpha(q) - h(q)] + 1
        return alpha, f_alpha

    @staticmethod
    def mfdfa_rolling(series, window_size, q_list, scales, order=1):
        """
        Applique MF-DFA sur des fenêtres glissantes de taille 'window_size'
        et renvoie la largeur du spectre multifractal (Delta alpha) pour chaque fenêtre.

        Paramètres:
            series      : pd.Series ou array-like, la série de données.
            window_size : int, nombre de points par fenêtre glissante.
            q_list      : liste/array des ordres q pour MF-DFA.
            scales      : liste/array d'échelles (tailles de segments).
            order       : ordre du polynôme pour le detrending (1 = linéaire par défaut).

        Retourne:
            pd.Series contenant la largeur du spectre Delta alpha pour chaque fenêtre,
            indexé par la date (ou l'index) correspondant à la fin de la fenêtre.
        """
        if isinstance(series, pd.Series):
            data = series.values
            index_data = series.index
        else:
            data = np.array(series)
            index_data = np.arange(len(data))

        alpha_widths = []
        rolling_index = []
        nb_points = len(data)

        for start in range(nb_points - window_size + 1):
            end = start + window_size
            window_data = data[start:end]

            # 1) Calcul de Fq(s) pour la fenêtre
            Fq = ComputeMFDFA.mfdfa(window_data, scales, q_list, order=order)

            # 2) Calcul de h(q) par régression linéaire de log(Fq) vs log(s)
            h_q = []
            log_scales = np.log(scales)
            for j, q in enumerate(q_list):
                log_Fq = np.log(Fq[j, :])
                # Ajustement linéaire pour trouver la pente = h(q)
                coeffs = np.polyfit(log_scales, log_Fq, 1)
                h_q.append(coeffs[0])
            h_q = np.array(h_q)

            # 3) Transformation de Legendre -> alpha, f(alpha)
            alpha, f_alpha = ComputeMFDFA.compute_alpha_falpha(q_list, h_q)

            # 4) Calcul de la largeur du spectre multifractal
            alpha_width = alpha.max() - alpha.min()
            alpha_widths.append(alpha_width)
            rolling_index.append(index_data[end - 1])

        return pd.Series(alpha_widths, index=rolling_index, name="alpha_width")

    @staticmethod
    def fa(x, scales, qs):
        """
        Fluctuation Analysis (FA) pour des séries stationnaires normalisées.

        Paramètres :
            x      : série temporelle (tableau 1D)
            scales : liste (ou tableau) des tailles d'échelle s.
                     On suppose que pour chaque s, N est un multiple entier de s.
            qs     : tableau des valeurs q (q peut être négatif, positif ou zéro).

        Retourne :
            Fqs    : tableau 2D de forme (len(qs), len(scales)) contenant Fq(s) pour chaque q et chaque échelle s.
        """
        x = np.array(x)
        N = len(x)

        # Calcul du profil avec Y(0)=0
        Y = np.zeros(N + 1)
        Y[1:] = np.cumsum(x - np.mean(x))
        Fqs = np.zeros((len(qs), len(scales)))

        for j, s in enumerate(scales):
            s = int(s)
            # On suppose que N est un multiple entier de s
            Ns = N // s
            F_loc = np.empty(Ns)
            for k in range(1, Ns + 1):
                F_loc[k - 1] = np.abs(Y[k * s] - Y[(k - 1) * s])
            for i, q in enumerate(qs):
                if q == 0:
                    Fqs[i, j] = np.exp(np.mean(np.log(F_loc)))
                else:
                    Fqs[i, j] = (np.mean(F_loc ** q)) ** (1 / q)
        return Fqs