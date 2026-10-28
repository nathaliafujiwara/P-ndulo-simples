#!/usr/bin/env python3
"""
Analisa um vídeo de pêndulo simples para gerar uma tabela (t, x, y) e calcular fator de qualidade Q.

Massa = 150 g (verde), fio = 60 cm (branco),
amplitude inicial de +30 cm a -30 cm (theta0 ~ 30 graus).

Método (robusto a câmera na mão e pivô fora do quadro):
  1. Detecta a massa verde (HSV) e o ponto onde o fio se prende nela.
  2. Mede o ÂNGULO do fio em cada quadro varrendo raios a partir da massa
     e escolhendo o de maior contraste (o fio é uma linha fina e clara).
  3. Com L = 60 cm:  x = L sin(theta),  y = -L cos(theta)  (origem no pivô).
  4. Q é calculado por três métodos independentes.

"""
import argparse
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

MASSA = 0.150      # kg
L = 0.60           # m
G = 9.81           # m/s^2

VERDE_MIN = np.array([40, 90, 40]) ##verde no HVS
VERDE_MAX = np.array([90, 255, 255])

ANGULOS = np.radians(np.arange(-45, 45.01, 0.25))   # raios testados (procurando o fio)
DIST = np.arange(50, 330, 2.0)                      # px ao longo do raio
SA, CA = np.sin(ANGULOS)[:, None], np.cos(ANGULOS)[:, None]


def achar_massa(frame_hsv):
    """Retorna (cx, cy, topo_xy) da massa verde, ou None."""
    m = cv2.inRange(frame_hsv, VERDE_MIN, VERDE_MAX)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: ##sem "mancha" verde
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 300: ##provável ruído
        return None
    mo = cv2.moments(c) 
    pts = c[:, 0, :]
    topo = pts[pts[:, 1] <= pts[:, 1].min() + 4].mean(0)
    return mo["m10"] / mo["m00"], mo["m01"] / mo["m00"], topo ##centro da "mancha" verde


def angulo_fio(gray, topo):
    """Ângulo (graus, + = fio inclinado p/ a direita acima da massa) e contraste."""
    def amostra(off):
        X = topo[0] + DIST * SA + off * CA
        Y = topo[1] - DIST * CA + off * SA
        return map_coordinates(gray, [Y, X], order=1, mode="nearest")
    score = (amostra(0) - 0.5 * (amostra(5) + amostra(-5))).mean(1)
    i = int(score.argmax())
    return float(np.degrees(ANGULOS[i])), float(score[i])


def rastrear(caminho, mostrar=False): ##percorre o vídeo
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        sys.exit("Não foi possível abrir o vídeo.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    linhas, n = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV) ##converte para HVS (facilita achar cores)
        r = achar_massa(hsv)
        if r is not None:
            cx, cy, topo = r
            gray = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
            ang, sc = angulo_fio(gray, topo)
            linhas.append((n / fps, cx, cy, ang))
            if mostrar:
                a = np.radians(ang)
                p2 = (int(topo[0] + 300 * np.sin(a)), int(topo[1] - 300 * np.cos(a)))
                cv2.line(f, (int(topo[0]), int(topo[1])), p2, (0, 255, 255), 2)
                cv2.circle(f, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.imshow("rastreamento (q sai)", f)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        n += 1
    cap.release()
    cv2.destroyAllWindows()
    return fps, np.array(linhas)


def amortecida(t, A, gamma, w, fase, c): ##para gerar um dos gráficos
    return A * np.exp(-gamma * t) * np.cos(w * t + fase) + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mostrar", action="store_true")
    ap.add_argument("--saida", default="pendulo_posicoes.csv") ##tabela
    ap.add_argument("--png", default="pendulo_resultados.png") ##gráficos
    a = ap.parse_args([r"C:\Users\Junior\OneDrive\Documentos\Nathi\vscodigando\WhatsApp Video 2026-10-28 at 09.02.54.mp4"]) ##vídeo
    fps, d = rastrear(a.video, a.mostrar) ##lê o vídeo
    if len(d) < 50:
        sys.exit("Poucos quadros com a massa detectada: ajuste VERDE_MIN/VERDE_MAX.")
    t, ang = d[:, 0], d[:, 3]
    print(f"{len(t)} quadros úteis, {fps:.2f} fps, duração {t[-1]-t[0]:.1f} s")

    # ---------- tabela ----------
    # offset de equilíbrio (câmera levemente inclinada): ajustado no fit abaixo
    th0 = np.radians(ang) ##ângulo
    dt0 = float(np.median(np.diff(t))) ##intervalo de tempo
    x = 100 * L * np.sin(th0)        # cm
    y = -100 * L * np.cos(th0)       # cm (pivô na origem)
    w_ = savgol_filter(th0, 9, 3, deriv=1, delta=dt0) ##velocidade angular
    # energia mecânica (referência: ponto mais baixo), usando theta suavizado
    E = MASSA * L**2 * w_**2 / 2 + MASSA * G * L * (1 - np.cos(th0))

    # ---------- Q ----------
    # (1) fit senoide amortecida em theta(t)
    dt = np.median(np.diff(t))
    esp = np.abs(np.fft.rfft(th0 - th0.mean()))
    fr = np.fft.rfftfreq(len(t), dt)
    w0g = 2 * np.pi * fr[esp[1:].argmax() + 1]
    p0 = [np.ptp(th0) / 2, 0.01, w0g, 0, th0.mean()]
    p, cov = curve_fit(amortecida, t, th0, p0=p0, maxfev=40000)
    A, gam, w, fase, c = p
    perr = np.sqrt(np.diag(cov))
    wn = np.hypot(w, gam)
    Q_fit = wn / (2 * gam)
    Q_fit_err = Q_fit * perr[1] / gam

    # (2) envelope: picos positivos e negativos |theta - c|
    pk, _ = find_peaks(th0 - c, distance=int(0.6 * fps * 2 * np.pi / w))
    vl, _ = find_peaks(-(th0 - c), distance=int(0.6 * fps * 2 * np.pi / w))
    tp = np.r_[t[pk], t[vl]]
    ap_ = np.degrees(np.r_[(th0 - c)[pk], -(th0 - c)[vl]])   # graus
    ok_ = tp > 1.0          # descarta o 1º pico (quadros iniciais, incompleto)
    tp, ap_ = tp[ok_], ap_[ok_]
    co, cv_ = np.polyfit(tp, np.log(ap_), 1, cov=True)
    gam_env = -co[0]
    gam_env_err = np.sqrt(cv_[0, 0])
    Tmed = np.median(np.diff(np.sort(t[pk[t[pk] > 1.0]])))
    w_med = 2 * np.pi / Tmed
    Q_env = w_med / (2 * gam_env)
    Q_env_err = Q_env * gam_env_err / gam_env

    # (3) energia: Q = 2*pi*E / (perda de E por ciclo), E ~ exp(-2 gamma t)
    cE = np.polyfit(t, np.log(E), 1)
    # E oscila dentro do ciclo (erro de medida da velocidade); usa-se a média móvel de 1 período
    k = max(1, int(round(Tmed * fps)))
    Em = np.convolve(E, np.ones(k) / k, mode="valid")
    tm = t[k // 2: k // 2 + len(Em)]
    cE = np.polyfit(tm, np.log(Em), 1)
    Q_E = w_med / (-cE[0])

    T_teo = 2 * np.pi * np.sqrt(L / G)
    print("\n=========== RESULTADOS ===========")
    print(f"Ângulo inicial (pico)       : {ang[:int(1.7*fps)].max():.1f}° / {ang[:int(1.7*fps)].min():.1f}°")
    print(f"Equilíbrio (offset câmera)  : {c:.2f}°")
    print(f"Período médio T             : {Tmed:.4f} s   (pequenas osc.: {T_teo:.4f} s)")
    print(f"gamma (fit)                 : {gam:.5f} ± {perr[1]:.5f} 1/s")
    print(f"gamma (envelope)            : {gam_env:.5f} ± {gam_env_err:.5f} 1/s")
    print(f"b = 2 m gamma               : {2*MASSA*gam_env:.5f} kg/s")
    print(f"Q (fit senoide amortecida)  : {Q_fit:.0f} ± {Q_fit_err:.0f}")
    print(f"Q (envelope dos picos)      : {Q_env:.0f} ± {Q_env_err:.0f}")
    print(f"Q (energia)                 : {Q_E:.0f}")
    print("==================================")

    tab = pd.DataFrame({"t (s)": t, "x (cm)": x, "y (cm)": y, ##coisas da tabela
                        "theta (graus)": ang, "E (mJ)": E * 1e3})
    tab.to_csv(a.saida, index=False, float_format="%.4f")
    print(tab.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nTabela completa ({len(tab)} linhas) em {a.saida}")

    # ---------- gráficos ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(t, x, ".", ms=2, label="x medido")
    ax[0].plot(t, 100 * L * np.sin(amortecida(t, *p)), "r-", lw=0.8, label="ajuste")
    ax[0].set(xlabel="t (s)", ylabel="x (cm)", title="Posição horizontal")
    ax[0].legend()

    ax[1].semilogy(tp, ap_, "o", ms=3, label="picos |θ−c|")
    tt = np.linspace(t[0], t[-1], 100)
    ax[1].semilogy(tt, np.exp(np.polyval(co, tt)), "r-", label=f"Q ≈ {Q_env:.0f}")
    ax[1].set(xlabel="t (s)", ylabel="amplitude (°)", title="Envelope")
    ax[1].legend()
    
    ax[2].plot(x, y, ".", ms=2)
    ax[2].set(xlabel="x (cm)", ylabel="y (cm)", title="Trajetória (pivô na origem)", aspect="equal")
    plt.tight_layout()
    plt.savefig(a.png, dpi=140)


if __name__ == "__main__":
    main()