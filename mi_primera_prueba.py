import numpy as np
from qampy import equalisation, signals, impairments, helpers

fb = 40.e9
os = 2
fs = os*fb
N = 4*10**5
mu = 4e-4
t_pmd = 75e-12
theta2 = np.pi/2.1
M = 4
ntaps = 40
snr = 15

print("Generando señal QAM...")
sig = signals.SignalQAMGrayCoded(M, N, fb=fb, nmodes=2, dtype=np.complex128)
S = sig.resample(fs, renormalise=True, beta=0.1)
S = impairments.change_snr(S, snr)

print("Aplicando PMD (impairment)...")
SS = impairments.apply_PMD(S, theta2, t_pmd)

print("Entrenando ecualizador MCMA...")
wxy_m, err_m = equalisation.equalise_signal(SS, mu, Ntaps=ntaps, method="mcma", adaptive_stepsize=True)

print("Aplicando filtro...")
E_m = equalisation.apply_filter(SS, wxy_m)
E_m = helpers.normalise_and_center(E_m)

print("¡Ecualización completada exitosamente!")
print("Forma de los taps del ecualizador:", wxy_m.shape)
print("Forma de la señal ecualizada:", E_m.shape)
