import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

print("Démarrage : chargement de pandas (peut prendre quelques secondes)...", flush=True)
import numpy as np
import pandas as pd
 
# ---------------- Simulation parameters (document these in the report) ----
SEED = 2026
PERIOD_START = "2017-01"
PERIOD_END = "2018-08"
SAFETY_MONTHS = 0.5            # safety stock, in months of average demand
COVERAGE_MONTHS = 2.0          # each supplier order covers 2 months of demand
UNIT_COST_RATIO = 0.60         # purchase cost = 60 % of average selling price
PROMISED_LEAD_DAYS = [5, 7, 10, 14]       # possible contractual lead times
RELIABILITY_BETA = (8, 2)      # on-time probability ~ Beta(8,2) → mean 80 %
LATE_DELAY_MEAN_DAYS = 6       # extra days when a delivery is late (Poisson)
PARTIAL_PROB = 0.08            # probability of a partial delivery
PARTIAL_RANGE = (0.5, 0.95)    # share received when partial
# --------------------------------------------------------------------------
 
T0 = time.time()
_STEP = {"n": 0}
TOTAL_STEPS = 6
 
 
def progress(msg):
    """Print a numbered step with elapsed time, immediately (flush)."""
    _STEP["n"] += 1
    print(f"[{_STEP['n']}/{TOTAL_STEPS}] {msg:<55} (+{time.time() - T0:5.1f} s)", flush=True)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    pbi = Path(args.out) / "powerbi"
    rng = np.random.default_rng(SEED)
 
    progress("Chargement des tables produites par le script 02")
    ventes = pd.read_csv(pbi / "fact_ventes.csv", parse_dates=["date_achat"])
    produits = pd.read_csv(pbi / "dim_produit.csv")
    fournisseurs = pd.read_csv(pbi / "dim_fournisseur.csv")
 
    months = pd.period_range(PERIOD_START, PERIOD_END, freq="M")
    M = len(months)
    ventes = ventes[ventes["commande_valide"] == 1].copy()
    ventes["mois"] = ventes["date_achat"].dt.to_period("M")
 
    progress("Calcul de la demande mensuelle par produit")
    # ---- Demand matrix: products x months (real Olist sales) ----
    demand_df = (ventes.groupby(["product_id", "mois"])["quantite"].sum()
                       .unstack(fill_value=0)
                       .reindex(columns=months, fill_value=0))
    prod_ids = demand_df.index.to_numpy()
    P = len(prod_ids)
    demand = demand_df.to_numpy(dtype=int)
 
    prod = produits.set_index("product_id").reindex(prod_ids)
    supplier = prod["fournisseur_principal"].to_numpy()
    unit_cost = (prod["prix_moyen"].to_numpy() * UNIT_COST_RATIO).round(2)
 
    print(f"      {P:,} produits x {M} mois", flush=True)
    progress("Tirage des paramètres fournisseurs")
    # ---- Supplier parameters ----
    sellers = fournisseurs["seller_id"].to_numpy()
    params = pd.DataFrame({
        "seller_id": sellers,
        "delai_promis_j": rng.choice(PROMISED_LEAD_DAYS, size=len(sellers)),
        "fiabilite": rng.beta(*RELIABILITY_BETA, size=len(sellers)).round(3),
        "source": "simulé",
    })
    sp = params.set_index("seller_id").reindex(supplier)
    lead = sp["delai_promis_j"].fillna(10).to_numpy()
    reliab = sp["fiabilite"].fillna(0.8).to_numpy()
 
    # ---- Replenishment policy ----
    avg_d = demand.mean(axis=1)
    active = avg_d > 0
    rop = np.where(active, np.ceil(avg_d * (lead / 30 + SAFETY_MONTHS)), 0).astype(int)
    qty = np.where(active, np.maximum(np.ceil(avg_d * COVERAGE_MONTHS), 1), 0).astype(int)
    initial = rop + qty
 
    # receipts[p, m] = units arriving during month m (extra columns = after the period)
    receipts = np.zeros((P, M + 6), dtype=int)
    on_order = np.zeros(P, dtype=int)
    month_starts = months.to_timestamp()
    month_ends = months.to_timestamp(how="end").normalize()
 
    stock_rows = []
    po_rows = []
    opening = initial.copy()
    po_counter = 0
 
    progress("Simulation mois par mois")
    for m in range(M):
        rec = receipts[:, m]
        on_order -= rec
        available = opening + rec
        sold = np.minimum(available, demand[:, m])
        unmet = demand[:, m] - sold
        closing = available - sold
        days_out = np.where(demand[:, m] > 0, np.round(30 * unmet / np.maximum(demand[:, m], 1)), 0)
 
        stock_rows.append(pd.DataFrame({
            "product_id": prod_ids,
            "seller_id": supplier,
            "mois": month_starts[m].date(),
            "stock_initial": opening,
            "quantite_recue": rec,
            "demande": demand[:, m],
            "quantite_vendue": sold,
            "quantite_non_servie": unmet,
            "stock_final": closing,
            "en_rupture": (unmet > 0).astype(int),
            "jours_rupture_estimes": days_out.astype(int),
            "point_de_commande": rop,
            "cout_unitaire": unit_cost,
            "valeur_stock_final": (closing * unit_cost).round(2),
        }))
 
        # ---- Reorder decision at end of month ----
        need = active & (closing + on_order <= rop)
        idx = np.flatnonzero(need)
        if len(idx):
            order_date = month_ends[m]
            on_time = rng.random(len(idx)) < reliab[idx]
            delay = np.where(on_time, 0, 1 + rng.poisson(LATE_DELAY_MEAN_DAYS, len(idx)))
            partial = rng.random(len(idx)) < PARTIAL_PROB
            share = np.where(partial, rng.uniform(*PARTIAL_RANGE, len(idx)), 1.0)
            q_ord = qty[idx]
            q_rec = np.maximum(np.floor(q_ord * share), 1).astype(int)
 
            promised = order_date + pd.to_timedelta(lead[idx], unit="D")
            actual = promised + pd.to_timedelta(delay, unit="D")
            arr_month = ((actual.year - month_starts[0].year) * 12
                         + actual.month - month_starts[0].month).to_numpy()
            arr_month = np.clip(arr_month, m + 1, receipts.shape[1] - 1)
            np.add.at(receipts, (idx, arr_month), q_rec)
            on_order[idx] += q_rec
 
            po_rows.append(pd.DataFrame({
                "po_id": [f"PO{po_counter + i + 1:07d}" for i in range(len(idx))],
                "seller_id": supplier[idx],
                "product_id": prod_ids[idx],
                "date_commande": order_date.date(),
                "date_livraison_prevue": promised.date,
                "date_livraison_reelle": actual.date,
                "delai_promis_j": lead[idx].astype(int),
                "delai_reel_j": (lead[idx] + delay).astype(int),
                "retard_j": delay.astype(int),
                "livree_a_temps": on_time.astype(int),
                "quantite_commandee": q_ord,
                "quantite_recue": q_rec,
                "livraison_complete": (q_rec == q_ord).astype(int),
                "cout_unitaire": unit_cost[idx],
                "cout_total": (q_rec * unit_cost[idx]).round(2),
            }))
            po_counter += len(idx)
 
        opening = closing
        print(f"      mois {m + 1:>2}/{M} ({months[m]}) : {int((unmet > 0).sum()):>6,} produits en rupture, "
              f"{int(need.sum()):>6,} commandes fournisseurs passées", flush=True)
 
    progress("Assemblage des résultats")
    stock = pd.concat(stock_rows, ignore_index=True)
    stock["source"] = "simulé"
    # keep only months where the product exists in the catalogue (from its first sale)
    first_sale = ventes.groupby("product_id")["mois"].min().dt.to_timestamp().dt.date
    stock = stock[pd.to_datetime(stock["mois"]) >= pd.to_datetime(stock["product_id"].map(first_sale))]
 
    po = pd.concat(po_rows, ignore_index=True) if po_rows else pd.DataFrame()
    if len(po):
        period_end = month_ends[-1].date()
        po["statut"] = np.where(pd.to_datetime(po["date_livraison_reelle"]) > pd.Timestamp(period_end),
                                "en_cours", "livree")
        po["source"] = "simulé"
 
    progress("Export des fichiers CSV")
    stock.to_csv(pbi / "fact_stock_mensuel.csv", index=False, encoding="utf-8-sig")
    po.to_csv(pbi / "fact_livraisons_fournisseurs.csv", index=False, encoding="utf-8-sig")
    params.to_csv(pbi / "parametres_fournisseurs.csv", index=False, encoding="utf-8-sig")
 
    print(f"fact_stock_mensuel.csv            {len(stock):>9,} lignes")
    print(f"fact_livraisons_fournisseurs.csv  {len(po):>9,} lignes")
    print(f"parametres_fournisseurs.csv       {len(params):>9,} lignes")
    print("\n=== Contrôle rapide (données simulées) ===")
    active_rows = stock[stock["demande"] > 0]
    print(f"Taux de rupture (mois-produit avec demande) : {100 * active_rows['en_rupture'].mean():.1f} %")
    if len(po):
        print(f"Livraisons fournisseurs à temps            : {100 * po['livree_a_temps'].mean():.1f} %")
        print(f"Livraisons complètes                       : {100 * po['livraison_complete'].mean():.1f} %")
    print(f"\nTerminé   (durée totale : {time.time() - T0:.1f} s)")
 
 
if __name__ == "__main__":
    main()
 