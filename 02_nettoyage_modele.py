import argparse
import sys
import time
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

print("Démarrage : chargement de pandas (peut prendre quelques secondes)...", flush=True)
import numpy as np
import pandas as pd
 
PERIOD_START = pd.Timestamp("2017-01-01")
PERIOD_END = pd.Timestamp("2018-08-31 23:59:59")
 
# Two categories are missing from Olist's translation table
MISSING_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}
INVALID_STATUSES = {"canceled", "unavailable"}
 
MOIS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
           "août", "septembre", "octobre", "novembre", "décembre"]
JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
 
log = []
 
T0 = time.time()
_STEP = {"n": 0}
TOTAL_STEPS = 11
 
 
def progress(msg):
    """Print a numbered step with elapsed time, immediately (flush)."""
    _STEP["n"] += 1
    print(f"[{_STEP['n']}/{TOTAL_STEPS}] {msg:<55} (+{time.time() - T0:5.1f} s)", flush=True)
 
 
 
 
def journal(table, traitement, avant, apres, justification=""):
    log.append({"table": table, "traitement": traitement, "lignes_avant": avant,
                "lignes_apres": apres, "lignes_impactees": avant - apres if avant != apres else "",
                "justification": justification})
 
 
def normalize_text(s: pd.Series) -> pd.Series:
    def _n(x):
        if pd.isna(x):
            return x
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return " ".join(x.lower().split())
    return s.map(_n)
 
 
def read(data_dir, fname, dates=()):
    print(f"      lecture : {fname:<42}", end="", flush=True)
    t = time.time()
    df = pd.read_csv(data_dir / fname, encoding="utf-8-sig")
    print(f"{len(df):>9,} lignes  ({time.time() - t:.1f} s)", flush=True)
    for col in dates:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
 
 
def build_dim_date(start, end):
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    dd = pd.DataFrame({"date": dates})
    dd["annee"] = dd["date"].dt.year
    dd["trimestre"] = "T" + dd["date"].dt.quarter.astype(str)
    dd["mois_num"] = dd["date"].dt.month
    dd["mois"] = dd["mois_num"].map(lambda m: MOIS_FR[m - 1])
    dd["annee_mois"] = dd["date"].dt.strftime("%Y-%m")
    dd["debut_mois"] = dd["date"].dt.to_period("M").dt.to_timestamp()
    dd["jour_semaine_num"] = dd["date"].dt.weekday + 1
    dd["jour_semaine"] = dd["date"].dt.weekday.map(lambda w: JOURS_FR[w])
    dd["week_end"] = (dd["date"].dt.weekday >= 5).astype(int)
    return dd
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out) / "powerbi"
    out_dir.mkdir(parents=True, exist_ok=True)
 
    progress("Chargement des fichiers")
    o = read(data_dir, "olist_orders_dataset.csv",
             ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"])
    c = read(data_dir, "olist_customers_dataset.csv")
    it = read(data_dir, "olist_order_items_dataset.csv", ["shipping_limit_date"])
    p = read(data_dir, "olist_order_payments_dataset.csv")
    r = read(data_dir, "olist_order_reviews_dataset.csv",
             ["review_creation_date", "review_answer_timestamp"])
    pr = read(data_dir, "olist_products_dataset.csv")
    s = read(data_dir, "olist_sellers_dataset.csv")
    cat = read(data_dir, "product_category_name_translation.csv")
 
    # ------------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------------
    progress("Nettoyage des commandes")
    n0 = len(o)
    o = o.drop_duplicates(subset="order_id")
    journal("orders", "Suppression des commandes en double", n0, len(o))
 
    n0 = len(o)
    o = o[o["order_purchase_timestamp"].between(PERIOD_START, PERIOD_END)].copy()
    journal("orders", "Restriction à la période 2017-01 → 2018-08", n0, len(o),
            "Mois de début et de fin d'extraction incomplets (voir rapport qualité)")
 
    n0 = len(o)
    o = o[o["customer_id"].isin(c["customer_id"])]
    journal("orders", "Exclusion des commandes avec client inconnu", n0, len(o))
 
    buy = o["order_purchase_timestamp"]
    for col, label in [("order_approved_at", "validation"),
                       ("order_delivered_carrier_date", "remise transporteur"),
                       ("order_delivered_customer_date", "livraison client")]:
        bad = o[col] < buy
        o.loc[bad, col] = pd.NaT
        journal("orders", f"Date de {label} antérieure à l'achat → mise à vide",
                len(o), len(o), f"{int(bad.sum())} dates corrigées")
    bad = o["order_delivered_customer_date"] < o["order_delivered_carrier_date"]
    o.loc[bad, "order_delivered_carrier_date"] = pd.NaT
    journal("orders", "Remise transporteur postérieure à la livraison → mise à vide",
            len(o), len(o), f"{int(bad.sum())} dates corrigées")
 
    # ------------------------------------------------------------------
    # CUSTOMERS  (customer_id = one per order ; customer_unique_id = real person)
    # ------------------------------------------------------------------
    progress("Nettoyage des clients")
    c["customer_city"] = normalize_text(c["customer_city"])
    journal("customers", "Normalisation des noms de ville (minuscules, sans accents)", len(c), len(c))
    o = o.merge(c[["customer_id", "customer_unique_id", "customer_city", "customer_state"]],
                on="customer_id", how="left")
    journal("customers", "Remplacement de customer_id par customer_unique_id comme identifiant client",
            c["customer_id"].nunique(), c["customer_unique_id"].nunique(),
            "Un client réel possède un customer_id différent à chaque commande")
 
    # ------------------------------------------------------------------
    # PRODUCTS / CATEGORIES
    # ------------------------------------------------------------------
    progress("Nettoyage des produits et catégories")
    tr = dict(zip(cat["product_category_name"], cat["product_category_name_english"]))
    tr.update(MISSING_TRANSLATIONS)
    n_null = int(pr["product_category_name"].isna().sum())
    pr["product_category_name"] = pr["product_category_name"].fillna("non_renseignee")
    tr["non_renseignee"] = "not_specified"
    pr["categorie"] = pr["product_category_name"].map(tr).fillna(pr["product_category_name"])
    journal("products", "Catégorie manquante → 'non_renseignee'", len(pr), len(pr),
            f"{n_null} produits concernés")
    journal("categories", "Ajout des traductions manquantes", len(cat), len(cat) + len(MISSING_TRANSLATIONS),
            ", ".join(MISSING_TRANSLATIONS))
 
    # ------------------------------------------------------------------
    # ORDER ITEMS
    # ------------------------------------------------------------------
    progress("Nettoyage des articles vendus")
    n0 = len(it)
    it = it.drop_duplicates(subset=["order_id", "order_item_id"])
    it = it[it["order_id"].isin(o["order_id"])]
    journal("order_items", "Conservation des articles des commandes retenues", n0, len(it))
    n0 = len(it)
    it = it[it["product_id"].isin(pr["product_id"]) & it["seller_id"].isin(s["seller_id"])]
    journal("order_items", "Exclusion des articles avec produit ou vendeur inconnu", n0, len(it))
    n0 = len(it)
    it = it[(it["price"] > 0) & (it["freight_value"] >= 0)]
    journal("order_items", "Exclusion des prix ≤ 0 et frais négatifs", n0, len(it))
 
    # ------------------------------------------------------------------
    # PAYMENTS (aggregated per order)
    # ------------------------------------------------------------------
    progress("Agrégation des paiements")
    n0 = len(p)
    p = p.drop_duplicates(subset=["order_id", "payment_sequential"])
    p = p[p["order_id"].isin(o["order_id"])].copy()
    p["payment_type"] = p["payment_type"].replace("not_defined", "non_renseigne")
    journal("payments", "Paiements des commandes retenues ; 'not_defined' → 'non_renseigne'", n0, len(p))
    main_type = (p.sort_values("payment_value", ascending=False)
                  .drop_duplicates("order_id")[["order_id", "payment_type"]])
    pay = (p.groupby("order_id")
             .agg(montant_paye=("payment_value", "sum"),
                  nb_versements=("payment_installments", "max"),
                  nb_moyens_paiement=("payment_type", "nunique"))
             .reset_index()
             .merge(main_type, on="order_id", how="left")
             .rename(columns={"payment_type": "mode_paiement"}))
 
    # ------------------------------------------------------------------
    # REVIEWS (one per order: the most recent)
    # ------------------------------------------------------------------
    progress("Dédoublonnage des avis")
    n0 = len(r)
    r = r[r["order_id"].isin(o["order_id"]) & r["review_score"].between(1, 5)]
    r = (r.sort_values(["review_answer_timestamp", "review_creation_date"])
          .drop_duplicates("order_id", keep="last"))
    journal("reviews", "Un seul avis par commande (le plus récent)", n0, len(r),
            "Certaines commandes avaient plusieurs avis")
    r = r.assign(
        avis_avec_commentaire=r["review_comment_message"].notna().astype(int),
        delai_reponse_avis_h=((r["review_answer_timestamp"] - r["review_creation_date"])
                              .dt.total_seconds() / 3600).round(1),
    )[["order_id", "review_score", "avis_avec_commentaire", "delai_reponse_avis_h"]]
 
    # ------------------------------------------------------------------
    # FACT_COMMANDES (order grain)
    # ------------------------------------------------------------------
    progress("Construction de fact_commandes")
    items_agg = (it.groupby("order_id")
                   .agg(nb_articles=("order_item_id", "count"),
                        montant_articles=("price", "sum"),
                        frais_livraison=("freight_value", "sum"),
                        nb_fournisseurs=("seller_id", "nunique"))
                   .reset_index())
    fc = (o.merge(items_agg, on="order_id", how="left")
           .merge(pay, on="order_id", how="left")
           .merge(r, on="order_id", how="left"))
 
    delivered_ok = fc["order_status"].eq("delivered") & fc["order_delivered_customer_date"].notna()
    delivered_date = fc["order_delivered_customer_date"].dt.normalize()
    est_date = fc["order_estimated_delivery_date"].dt.normalize()
    fc["delai_validation_h"] = ((fc["order_approved_at"] - fc["order_purchase_timestamp"])
                                .dt.total_seconds() / 3600).round(1)
    fc["delai_livraison_j"] = np.where(
        delivered_ok,
        (fc["order_delivered_customer_date"] - fc["order_purchase_timestamp"]).dt.total_seconds() / 86400,
        np.nan).round(1)
    fc["retard_j"] = np.where(delivered_ok, (delivered_date - est_date).dt.days, np.nan)
    fc["livree_a_temps"] = np.where(delivered_ok, (fc["retard_j"] <= 0).astype(float), np.nan)
    fc["commande_valide"] = (~fc["order_status"].isin(INVALID_STATUSES)).astype(int)
    fc["ecart_paiement"] = (fc["montant_paye"]
                            - fc["montant_articles"].fillna(0) - fc["frais_livraison"].fillna(0)).round(2) + 0.0
 
    fact_commandes = pd.DataFrame({
        "order_id": fc["order_id"],
        "client_id": fc["customer_unique_id"],
        "date_achat": fc["order_purchase_timestamp"].dt.date,
        "heure_achat": fc["order_purchase_timestamp"].dt.hour,
        "statut": fc["order_status"],
        "commande_valide": fc["commande_valide"],
        "date_validation": fc["order_approved_at"].dt.date,
        "date_expedition": fc["order_delivered_carrier_date"].dt.date,
        "date_livraison": fc["order_delivered_customer_date"].dt.date,
        "date_livraison_estimee": fc["order_estimated_delivery_date"].dt.date,
        "delai_validation_h": fc["delai_validation_h"],
        "delai_livraison_j": fc["delai_livraison_j"],
        "retard_j": fc["retard_j"],
        "livree_a_temps": fc["livree_a_temps"],
        "nb_articles": fc["nb_articles"].fillna(0).astype(int),
        "nb_fournisseurs": fc["nb_fournisseurs"].fillna(0).astype(int),
        "montant_articles": fc["montant_articles"].fillna(0).round(2),
        "frais_livraison": fc["frais_livraison"].fillna(0).round(2),
        "montant_paye": fc["montant_paye"].round(2),
        "ecart_paiement": fc["ecart_paiement"],
        "mode_paiement": fc["mode_paiement"].fillna("aucun_paiement"),
        "nb_versements": fc["nb_versements"],
        "note_avis": fc["review_score"],
        "avis_avec_commentaire": fc["avis_avec_commentaire"],
        "delai_reponse_avis_h": fc["delai_reponse_avis_h"],
        "etat_client": fc["customer_state"],
    })
 
    # ------------------------------------------------------------------
    # FACT_VENTES (item grain)
    # ------------------------------------------------------------------
    progress("Construction de fact_ventes")
    fv = it.merge(fc[["order_id", "customer_unique_id", "order_purchase_timestamp",
                      "order_status", "commande_valide", "livree_a_temps"]],
                  on="order_id", how="left")
    fact_ventes = pd.DataFrame({
        "order_id": fv["order_id"],
        "ligne": fv["order_item_id"],
        "product_id": fv["product_id"],
        "seller_id": fv["seller_id"],
        "client_id": fv["customer_unique_id"],
        "date_achat": fv["order_purchase_timestamp"].dt.date,
        "statut": fv["order_status"],
        "commande_valide": fv["commande_valide"],
        "livree_a_temps": fv["livree_a_temps"],
        "quantite": 1,  # Olist: one row per unit sold
        "prix": fv["price"].round(2),
        "frais_livraison": fv["freight_value"].round(2),
    })
 
    # ------------------------------------------------------------------
    # DIMENSIONS
    # ------------------------------------------------------------------
    progress("Construction des dimensions")
    valid_orders = fc[fc["commande_valide"] == 1]
    cli_stats = (valid_orders.groupby("customer_unique_id")
                   .agg(date_premier_achat=("order_purchase_timestamp", "min"),
                        date_dernier_achat=("order_purchase_timestamp", "max"),
                        nb_commandes=("order_id", "nunique")))
    last_loc = (fc.sort_values("order_purchase_timestamp")
                  .drop_duplicates("customer_unique_id", keep="last")
                  .set_index("customer_unique_id")[["customer_city", "customer_state"]])
    n_ids = c.groupby("customer_unique_id")["customer_id"].nunique().rename("nb_identifiants_crm")
    dim_client = (last_loc.join(cli_stats, how="left").join(n_ids, how="left")
                  .reset_index()
                  .rename(columns={"customer_unique_id": "client_id",
                                   "customer_city": "ville", "customer_state": "etat"}))
    dim_client["nb_commandes"] = dim_client["nb_commandes"].fillna(0).astype(int)
    dim_client["type_client"] = np.select(
        [dim_client["nb_commandes"] >= 2, dim_client["nb_commandes"] == 1],
        ["Récurrent", "Nouveau"], "Sans commande valide")
    for col in ["date_premier_achat", "date_dernier_achat"]:
        dim_client[col] = dim_client[col].dt.date
 
    pr_used = pr[pr["product_id"].isin(fact_ventes["product_id"])].copy()
    main_supplier = (fact_ventes.groupby(["product_id", "seller_id"]).size()
                       .reset_index(name="n")
                       .sort_values(["product_id", "n"], ascending=[True, False])
                       .drop_duplicates("product_id")
                       .set_index("product_id")["seller_id"]
                       .rename("fournisseur_principal"))
    prod_stats = (fact_ventes.groupby("product_id").agg(prix_moyen=("prix", "mean"))
                    .join(main_supplier))
    dim_produit = pd.DataFrame({
        "product_id": pr_used["product_id"],
        "categorie_pt": pr_used["product_category_name"],
        "categorie": pr_used["categorie"],
        "poids_g": pr_used["product_weight_g"],
        "volume_cm3": (pr_used["product_length_cm"] * pr_used["product_height_cm"]
                       * pr_used["product_width_cm"]),
        "nb_photos": pr_used["product_photos_qty"],
    }).merge(prod_stats.round(2).reset_index(), on="product_id", how="left")
 
    s["seller_city"] = normalize_text(s["seller_city"])
    sel_stats = (fact_ventes.groupby("seller_id")
                   .agg(date_premiere_vente=("date_achat", "min"),
                        nb_produits=("product_id", "nunique"),
                        nb_articles_vendus=("order_id", "count")))
    dim_fournisseur = (s[s["seller_id"].isin(fact_ventes["seller_id"])]
                       .rename(columns={"seller_city": "ville", "seller_state": "etat",
                                        "seller_zip_code_prefix": "code_postal"})
                       .merge(sel_stats.reset_index(), on="seller_id", how="left"))
 
    date_max = max(fc["order_estimated_delivery_date"].max(),
                   fc["order_delivered_customer_date"].max(), PERIOD_END)
    dim_date = build_dim_date(PERIOD_START, date_max + pd.DateOffset(months=3))
 
    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------
    progress("Export des fichiers CSV")
    outputs = {
        "dim_date": dim_date, "dim_client": dim_client, "dim_produit": dim_produit,
        "dim_fournisseur": dim_fournisseur, "fact_commandes": fact_commandes,
        "fact_ventes": fact_ventes, "journal_nettoyage": pd.DataFrame(log),
    }
    print("      Fichiers écrits dans", out_dir)
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"      {name + '.csv':<26} {len(df):>9,} lignes", flush=True)
 
    print("\n=== Journal de nettoyage ===")
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(pd.DataFrame(log)[["table", "traitement", "lignes_avant", "lignes_apres"]]
              .to_string(index=False))
    print(f"\nTerminé   (durée totale : {time.time() - T0:.1f} s)")
 
 
if __name__ == "__main__":
    main()
 