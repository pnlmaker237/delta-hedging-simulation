"""
Delta-hedging d'un call vanille (Black-Scholes) sur 21 jours de bourse.

Convention de la fiche : position couverte = long call + (-Delta x 100 x 100) actions.
PnL_t = (C_t - C_{t-1}) + Delta_{t-1} (S_{t-1} - S_t) + r (Delta_{t-1} S_{t-1} - C_{t-1}) dt
(tout est multiplié par le notionnel : 100 contrats x 100 actions = 10 000 actions)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # retirer cette ligne pour afficher les graphiques à l'écran
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------- Paramètres -----------------------------
S0, K, R, SIGMA = 100.0, 100.0, 0.02, 0.20
T_TOTAL = 1.0                 # maturité 1 an
DAYS = 21                     # jours de trading simulés
DT = 1 / 252
NOTIONAL = 100 * 100          # 100 contrats x 100 actions = 10 000 actions
S_FLOOR = 0.01                # plancher du sous-jacent
T_FLOOR = 1e-8                # plancher du temps restant


# ------------------------ 1. Pricing et Greeks ------------------------
def bs_greeks(S, K, r, T, sigma):
    """Retourne (prix, delta, gamma, vega) d'un call européen Black-Scholes (q = 0)."""
    S = max(S, S_FLOOR)                        # cas limite : S -> 0
    if T <= T_FLOOR:                           # cas limite : T -> 0 : valeur intrinsèque
        return max(S - K, 0.0), float(S > K), 0.0, 0.0

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrtT)
    vega = S * sqrtT * norm.pdf(d1)            # par unité de vol (1.0 = 100 %)
    return price, delta, gamma, vega


# --------------------------- 2. Simulation ---------------------------
def simulate_stock(S0, r, sigma, T, days, rng=None):
    """Trajectoire GBM : S_t = S_{t-1} exp((r - sigma^2/2) dt + sigma sqrt(dt) Z)."""
    rng = np.random.default_rng() if rng is None else rng
    dt = DT
    Z = rng.standard_normal(days)
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return np.concatenate(([S0], S0 * np.exp(np.cumsum(increments))))  # taille days+1


# ------------------------- 3. Boucle de hedging -------------------------
def delta_hedge(S, K=K, r=R, sigma=SIGMA, T=T_TOTAL, notional=NOTIONAL):
    """Applique la stratégie sur une trajectoire S (taille days+1). Retourne un dict de tableaux."""
    n = len(S)
    C = np.zeros(n); delta = np.zeros(n); gamma = np.zeros(n); vega = np.zeros(n)
    pnl = np.zeros(n)
    position = np.zeros(n)                     # nombre d'actions détenues (négatif = short)

    # Jour 0 : prix de l'option, delta, mise en place de la couverture
    C[0], delta[0], gamma[0], vega[0] = bs_greeks(S[0], K, r, T, sigma)
    position[0] = -delta[0] * notional

    # Jours 1..N : nouveau S, mise à jour de l'option, PnL, rebalancement
    for t in range(1, n):
        T_rem = T - t * DT
        C[t], delta[t], gamma[t], vega[t] = bs_greeks(S[t], K, r, T_rem, sigma)

        pnl[t] = notional * (
            (C[t] - C[t - 1])                              # variation de valeur de l'option
            + delta[t - 1] * (S[t - 1] - S[t])             # gain/perte sur les actions shortées
            + r * (delta[t - 1] * S[t - 1] - C[t - 1]) * DT  # intérêts sur le cash
        )
        position[t] = -delta[t] * notional                 # rebalancement quotidien

    return dict(S=S, C=C, delta=delta, gamma=gamma, vega=vega, pnl=pnl,
                position=position, cum_pnl=np.cumsum(pnl))


# ------------------------------ 4. Résumé ------------------------------
def summary(res, label=""):
    print(f"{label:<28} PnL total = {res['cum_pnl'][-1]:>10,.0f} $ | "
          f"PnL quotidien moyen = {res['pnl'][1:].mean():>8,.0f} $ | "
          f"écart-type = {res['pnl'][1:].std():>8,.0f} $")


def print_table(res):
    print(f"{'Jour':>4} {'S':>8} {'C':>8} {'Delta':>7} {'Gamma':>8} {'Vega':>8} "
          f"{'Actions':>10} {'PnL ($)':>10}")
    for t in range(len(res["S"])):
        print(f"{t:>4} {res['S'][t]:>8.2f} {res['C'][t]:>8.3f} {res['delta'][t]:>7.3f} "
              f"{res['gamma'][t]:>8.4f} {res['vega'][t]:>8.2f} {res['position'][t]:>10,.0f} "
              f"{res['pnl'][t]:>10,.0f}")


# ------------------------------- Main -------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- Trajectoire principale, détail jour par jour ---
    S = simulate_stock(S0, R, SIGMA, T_TOTAL, DAYS, rng)
    main = delta_hedge(S)
    print("=== Trajectoire principale (sigma = 20 %) ===")
    print_table(main)
    print()
    summary(main, "Trajectoire principale")

    # --- 5 trajectoires (robustesse) ---
    print("\n=== 5 trajectoires, sigma = 20 % ===")
    paths = [delta_hedge(simulate_stock(S0, R, SIGMA, T_TOTAL, DAYS, rng)) for _ in range(5)]
    for i, p in enumerate(paths, 1):
        summary(p, f"Trajectoire {i}")

    # --- Test de robustesse : sigma = 10 % et 40 % ---
    print("\n=== Robustesse : PnL total moyen sur 200 trajectoires ===")
    for sig in (0.10, 0.20, 0.40):
        tot = [delta_hedge(simulate_stock(S0, R, sig, T_TOTAL, DAYS, rng), sigma=sig)["cum_pnl"][-1]
               for _ in range(200)]
        print(f"sigma = {sig:.0%} : moyenne = {np.mean(tot):>9,.0f} $ | "
              f"écart-type = {np.std(tot):>8,.0f} $ | pire = {np.min(tot):>9,.0f} $")

    # --- Graphiques ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    days_axis = np.arange(DAYS + 1)

    ax[0, 0].plot(days_axis, main["S"], label="Prix de l'action")
    ax[0, 0].axhline(K, color="grey", ls="--", label="Strike")
    ax[0, 0].set(title="Trajectoire simulée", xlabel="Jour", ylabel="Prix ($)")

    ax[0, 1].scatter(main["S"][1:], main["pnl"][1:], c=days_axis[1:], cmap="viridis")
    ax[0, 1].axhline(0, color="grey", lw=0.8)
    ax[0, 1].set(title="PnL quotidien vs S", xlabel="S ($)", ylabel="PnL ($)")

    ax[1, 0].plot(days_axis, main["delta"], label="Delta")
    ax[1, 0].plot(days_axis, main["gamma"] * 10, label="Gamma x10")
    ax[1, 0].set(title="Delta et Gamma", xlabel="Jour")

    ax[1, 1].plot(days_axis, main["cum_pnl"], label="Trajectoire principale")
    for p in paths:
        ax[1, 1].plot(days_axis, p["cum_pnl"], alpha=0.4)
    ax[1, 1].set(title="PnL cumulé (5 trajectoires + principale)", xlabel="Jour", ylabel="PnL ($)")

    for a in ax.ravel():
        a.grid(True)
    ax[0, 0].legend(); ax[1, 0].legend(); ax[1, 1].legend()
    plt.tight_layout()
    plt.savefig("delta_hedging.png", dpi=120)
    print("\nGraphique enregistré : delta_hedging.png")
