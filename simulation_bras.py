"""
Mini-projet 1GII - Sujet 34 : Commande en position d'un bras robotisé
Phase 4 : Simulation et commande
-----------------------------------------------------------------
Modèle :  J*theta'' + b*theta' = K*u - Td
   theta : angle du bras (rad)      u  : commande du moteur, limitée à [-1, 1]
   J : inertie   b : frottement     K  : gain du moteur   Td : couple perturbateur
Lance simplement :  python simulation_bras.py
(il faut numpy et matplotlib : pip install numpy matplotlib)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")           # pour enregistrer les figures sans fenêtre
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1) PARAMÈTRES (valeurs choisies, à justifier dans ton rapport)
# ---------------------------------------------------------------
J = 0.02      # inertie du bras            [kg.m^2]
b = 0.12      # frottement visqueux        [N.m.s/rad]
K = 0.6       # gain moteur (couple/unité de commande) [N.m]
U_MAX = 1.0   # saturation de la commande (tension max normalisée)
DT = 0.001    # pas de simulation [s]
T_FIN = 20.0  # durée [s]
t = np.arange(0, T_FIN, DT)

REF = np.deg2rad(45)   # consigne : 45 degrés

def sat(u):
    """Limite la commande entre -U_MAX et +U_MAX (le moteur a une limite)."""
    return max(-U_MAX, min(U_MAX, u))

# ---------------------------------------------------------------
# 2) FONCTION DE SIMULATION (méthode d'Euler, simple à comprendre)
# ---------------------------------------------------------------
def simuler(mode, Kp=1.0, Ki=0.0, u_const=0.0, Td_fn=lambda tt: 0.0):
    """mode = 'ouverte' (u constant) ou 'fermee' (correcteur P ou PI)."""
    theta, omega, integ = 0.0, 0.0, 0.0
    th, om, us = [], [], []
    for tt in t:
        e = REF - theta                      # erreur = consigne - mesure
        if mode == "ouverte":
            u = sat(u_const)
        else:
            u_brut = Kp * e + Ki * integ     # loi de commande P ou PI
            u = sat(u_brut)
            # anti-windup : on n'intègre que si le moteur n'est pas saturé
            if abs(u_brut) < U_MAX or (u_brut * e < 0):
                integ += e * DT
        # équation du bras : J*theta'' = K*u - b*theta' - Td
        acc = (K * u - b * omega - Td_fn(tt)) / J
        omega += acc * DT
        theta += omega * DT
        th.append(theta); om.append(omega); us.append(u)
    return np.rad2deg(np.array(th)), np.array(us)

def metriques(th):
    """Dépassement, temps de montée (10%->90%), erreur finale."""
    ref = np.rad2deg(REF)
    depass = max(0.0, (th.max() - ref) / ref * 100)
    try:
        t10 = t[np.argmax(th >= 0.1 * ref)]
        t90 = t[np.argmax(th >= 0.9 * ref)]
        tm = t90 - t10 if th.max() >= 0.9 * ref else float("nan")
    except Exception:
        tm = float("nan")
    err = ref - th[-int(1 / DT):].mean()
    return depass, tm, err

# ---------------------------------------------------------------
# 3) ESSAI 1 : BOUCLE OUVERTE (sans correcteur)
# ---------------------------------------------------------------
th_ouv, u_ouv = simuler("ouverte", u_const=0.3)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, th_ouv, lw=2, label="θ(t) avec u = 0,3 constant")
ax.axhline(45, color="r", ls="--", label="consigne 45°")
ax.set(xlabel="Temps (s)", ylabel="Angle (°)",
       title="Essai 1 - Boucle ouverte : le bras ne s'arrête jamais")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("fig1_boucle_ouverte.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------
# 4) ESSAIS 2 et 3 : BOUCLE FERMÉE P, comparaison de plusieurs Kp
# ---------------------------------------------------------------
gains = {0.1: "Kp = 0,1 (trop faible)", 1.0: "Kp = 1 (bon compromis)",
         4.0: "Kp = 4 (trop élevé)"}
res = {}
for kp in gains:
    res[kp] = simuler("fermee", Kp=kp)

fig, ax = plt.subplots(figsize=(8, 4.5))
for kp, lab in gains.items():
    ax.plot(t, res[kp][0], lw=2, label=lab)
ax.axhline(45, color="k", ls="--", label="consigne 45°")
ax.set(xlabel="Temps (s)", ylabel="Angle (°)", xlim=(0, 6),
       title="Essai 2 - Boucle fermée P : effet du gain Kp")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("fig2_comparaison_Kp.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for kp, lab in gains.items():
    ax.plot(t, res[kp][1], lw=2, label=lab)
ax.axhline(1, color="gray", ls=":"); ax.axhline(-1, color="gray", ls=":")
ax.set(xlabel="Temps (s)", ylabel="Commande u (normalisée)", xlim=(0, 6),
       title="Signal de commande u(t) - saturation à ±1")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("fig3_commande_saturation.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------
# 5) ESSAI 4 : PERTURBATION (couple résistant entre t=8s et t=14s)
#    Comparaison P  vs  PI
# ---------------------------------------------------------------
def perturbation(tt):
    return 0.15 if 8.0 <= tt < 14.0 else 0.0   # N.m (ex : poids d'un outil)

th_P, u_P = simuler("fermee", Kp=1.0, Td_fn=perturbation)
th_PI, u_PI = simuler("fermee", Kp=1.0, Ki=0.8, Td_fn=perturbation)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
a1.plot(t, th_P, lw=2, label="Correcteur P (Kp=1)")
a1.plot(t, th_PI, lw=2, label="Correcteur PI (Kp=1, Ki=0.8)")
a1.axhline(45, color="k", ls="--", label="consigne 45°")
a1.axvspan(8, 14, color="orange", alpha=.2, label="perturbation")
a1.set(ylabel="Angle (°)", title="Essai 4 - Rejet de perturbation")
a1.grid(alpha=.3); a1.legend(loc="lower right")
a2.plot(t, u_P, lw=2, label="u (P)"); a2.plot(t, u_PI, lw=2, label="u (PI)")
a2.axvspan(8, 14, color="orange", alpha=.2)
a2.set(xlabel="Temps (s)", ylabel="Commande u"); a2.grid(alpha=.3); a2.legend()
fig.tight_layout(); fig.savefig("fig4_perturbation_P_vs_PI.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------
# 6) TABLEAU DE RÉSULTATS (à copier dans ton rapport)
# ---------------------------------------------------------------
print("\n=== Performances (consigne 45°, sans perturbation) ===")
print(f"{'Réglage':<28}{'Dépassement %':>15}{'Tps montée (s)':>16}{'Erreur finale °':>17}")
for kp, lab in gains.items():
    d, tm, err = metriques(res[kp][0])
    print(f"{lab:<28}{d:>15.1f}{tm:>16.2f}{err:>17.2f}")

i_fin = int(13.9 / DT)
print("\n=== Erreur juste avant la fin de la perturbation (t=13,9 s) ===")
print(f"P  : {45 - th_P[i_fin]:.2f} °")
print(f"PI : {45 - th_PI[i_fin]:.2f} °")
print("\nFigures enregistrées : fig1 à fig4 (.png)")
