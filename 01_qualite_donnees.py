import argparse
import time
import unicodedata
from pathlib import Path
 
print("Démarrage : chargement de pandas (peut prendre quelques secondes)...", flush=True)
import numpy as np
import pandas as pd
 
FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "categories": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}
 
# Natural primary key of each table (None = no key)
PRIMARY_KEYS = {
    "orders": ["order_id"],
    "customers": ["customer_id"],
    "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"],
    "reviews": ["review_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "categories": ["product_category_name"],
    "geolocation": None,
}
 
DATE_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}
 
# Which "système" of the retail SI each table represents (for the report)
SYSTEME = {
    "orders": "POS / Ventes",
    "order_items": "POS / Ventes",
    "payments": "POS / Ventes",
    "customers": "CRM",
    "reviews": "CRM",
    "products": "Catalogue / Stock",
    "categories": "Catalogue / Stock",
    "sellers": "Fournisseurs",
    "geolocation": "Référentiel géographique",
}
 
 
T0 = time.time()
_STEP = {"n": 0}
TOTAL_STEPS = 8
 
 
def progress(msg):
    """Print a numbered step with elapsed time, immediately (flush)."""
    _STEP["n"] += 1
    print(f"[{_STEP['n']}/{TOTAL_STEPS}] {msg:<55} (+{time.time() - T0:5.1f} s)", flush=True)
 
 
def normalize_text(s: pd.Series) -> pd.Series:
    """Lowercase, remove accents and extra spaces (used to detect spelling variants)."""
    def _n(x):
        if pd.isna(x):
            return x
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return " ".join(x.lower().split())
    return s.map(_n)
 
 
def load(data_dir: Path):
    dfs, parse_errors = {}, []
    for i, (name, fname) in enumerate(FILES.items(), 1):
        path = data_dir / fname
        if not path.exists():
            raise SystemExit(f"Fichier manquant : {path}")
        print(f"      lecture {i}/{len(FILES)} : {fname:<42}", end="", flush=True)
        t = time.time()
        dfs[name] = pd.read_csv(path, encoding="utf-8-sig")
        print(f"{len(dfs[name]):>9,} lignes  ({time.time() - t:.1f} s)", flush=True)
    for name, cols in DATE_COLS.items():
        for c in cols:
            raw_not_null = dfs[name][c].notna().sum()
            dfs[name][c] = pd.to_datetime(dfs[name][c], errors="coerce")
            bad = raw_not_null - dfs[name][c].notna().sum()
            if bad:
                parse_errors.append((name, c, bad))
    return dfs, parse_errors
 
 
class Issues:
    def __init__(self):
        self.rows = []
 
    def add(self, dimension, table, controle, n, base, description, traitement):
        self.rows.append({
            "dimension_qualite": dimension,
            "table": table,
            "systeme": SYSTEME.get(table, ""),
            "controle": controle,
            "nb_cas": int(n),
            "base": int(base),
            "pourcentage": round(100 * n / base, 2) if base else 0.0,
            "description": description,
            "traitement_prevu": traitement,
        })
 
    def to_frame(self):
        return pd.DataFrame(self.rows)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dossier des CSV Olist")
    ap.add_argument("--out", default="output", help="dossier de sortie")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    progress("Chargement des fichiers")
    d, parse_errors = load(data_dir)
    o, c, it, p, r = d["orders"], d["customers"], d["order_items"], d["payments"], d["reviews"]
    pr, s, cat, g = d["products"], d["sellers"], d["categories"], d["geolocation"]
    iss = Issues()
 
    # ------------------------------------------------------------------
    # 1. Table overview + duplicates
    # ------------------------------------------------------------------
    progress("Doublons par table")
    overview = []
    for name, df in d.items():
        pk = PRIMARY_KEYS[name]
        dup_rows = int(df.duplicated().sum())
        dup_keys = int(df.duplicated(subset=pk).sum()) if pk else None
        overview.append({
            "table": name,
            "fichier": FILES[name],
            "systeme": SYSTEME[name],
            "nb_lignes": len(df),
            "nb_colonnes": df.shape[1],
            "cle_primaire": " + ".join(pk) if pk else "(aucune)",
            "lignes_dupliquees": dup_rows,
            "cles_dupliquees": dup_keys,
            "cellules_vides_%": round(100 * df.isna().sum().sum() / df.size, 2),
        })
        if dup_rows:
            iss.add("Unicité", name, "Lignes entièrement dupliquées", dup_rows, len(df),
                    "Mêmes valeurs sur toutes les colonnes",
                    "Supprimer les doublons exacts")
        if pk and dup_keys:
            iss.add("Unicité", name, f"Clé dupliquée ({' + '.join(pk)})", dup_keys, len(df),
                    "Une même clé apparaît plusieurs fois",
                    "Garder un seul enregistrement par clé")
    overview = pd.DataFrame(overview)
 
    # ------------------------------------------------------------------
    # 2. Missing values per column
    # ------------------------------------------------------------------
    progress("Valeurs manquantes")
    missing = []
    for name, df in d.items():
        for col in df.columns:
            n = int(df[col].isna().sum())
            missing.append({
                "table": name, "colonne": col, "type": str(df[col].dtype),
                "nb_manquants": n, "pct_manquants": round(100 * n / len(df), 2) if len(df) else 0,
            })
    missing = pd.DataFrame(missing).sort_values(["pct_manquants", "table"], ascending=[False, True])
 
    for name, col, bad in parse_errors:
        iss.add("Validité", name, f"Date illisible ({col})", bad, len(d[name]),
                "Format de date non reconnu", "Considérer comme valeur manquante")
 
    # ------------------------------------------------------------------
    # 3. Referential integrity (links between systems)
    # ------------------------------------------------------------------
    progress("Liens entre tables (intégrité référentielle)")
    def orphan(child, child_col, parent, parent_col, child_name, label, traitement):
        n = (~child[child_col].isin(parent[parent_col])).sum()
        iss.add("Intégrité référentielle", child_name, label, n, len(child),
                f"{child_col} absent de la table de référence", traitement)
 
    orphan(it, "order_id", o, "order_id", "order_items", "Articles sans commande", "Exclure")
    orphan(it, "product_id", pr, "product_id", "order_items", "Articles avec produit inconnu", "Exclure")
    orphan(it, "seller_id", s, "seller_id", "order_items", "Articles avec vendeur inconnu", "Exclure")
    orphan(o, "customer_id", c, "customer_id", "orders", "Commandes avec client inconnu", "Exclure")
    orphan(p, "order_id", o, "order_id", "payments", "Paiements sans commande", "Exclure")
    orphan(r, "order_id", o, "order_id", "reviews", "Avis sans commande", "Exclure")
 
    n = (~o["order_id"].isin(it["order_id"])).sum()
    iss.add("Complétude", "orders", "Commandes sans aucun article", n, len(o),
            "Commande enregistrée sans ligne de vente (souvent annulée/indisponible)",
            "Exclure des analyses de ventes")
    n = (~o["order_id"].isin(p["order_id"])).sum()
    iss.add("Complétude", "orders", "Commandes sans paiement", n, len(o),
            "Aucun paiement rattaché", "Signaler ; montant payé = vide")
    n = (~o["order_id"].isin(r["order_id"])).sum()
    iss.add("Complétude", "orders", "Commandes sans avis client", n, len(o),
            "Le client n'a pas laissé d'avis", "Normal ; exclure du calcul de satisfaction")
 
    # The same real person gets a new customer_id for every order -> CRM duplicates
    ids_per_person = c.groupby("customer_unique_id")["customer_id"].nunique()
    n = int((ids_per_person > 1).sum())
    iss.add("Unicité", "customers", "Clients physiques avec plusieurs identifiants",
            n, len(ids_per_person),
            "Un même client (customer_unique_id) possède plusieurs customer_id : "
            "pas de référentiel client unique",
            "Utiliser customer_unique_id comme identifiant client")
 
    # ------------------------------------------------------------------
    # 4. Date consistency (orders)
    # ------------------------------------------------------------------
    progress("Cohérence des dates")
    buy = o["order_purchase_timestamp"]
    checks = [
        ("Validation avant l'achat", o["order_approved_at"] < buy),
        ("Remise au transporteur avant l'achat", o["order_delivered_carrier_date"] < buy),
        ("Livraison client avant l'achat", o["order_delivered_customer_date"] < buy),
        ("Livraison client avant remise au transporteur",
         o["order_delivered_customer_date"] < o["order_delivered_carrier_date"]),
    ]
    for label, mask in checks:
        iss.add("Cohérence", "orders", label, mask.sum(), len(o),
                "Ordre chronologique des étapes non respecté",
                "Mettre la date incohérente à vide")
 
    delivered = o["order_status"].eq("delivered")
    iss.add("Cohérence", "orders", "Statut 'delivered' sans date de livraison",
            (delivered & o["order_delivered_customer_date"].isna()).sum(), delivered.sum(),
            "La commande est marquée livrée mais la date n'a pas été saisie",
            "Exclure du calcul des délais de livraison")
    iss.add("Cohérence", "orders", "Date de livraison renseignée mais statut ≠ 'delivered'",
            (~delivered & o["order_delivered_customer_date"].notna()).sum(), (~delivered).sum(),
            "Statut non mis à jour après la livraison", "Signaler")
    iss.add("Complétude", "orders", "Commandes non validées (order_approved_at vide)",
            o["order_approved_at"].isna().sum(), len(o),
            "Pas de date de validation", "Signaler")
 
    # ------------------------------------------------------------------
    # 5. Amounts and business rules
    # ------------------------------------------------------------------
    progress("Montants, avis, catalogue, villes, géolocalisation")
    iss.add("Validité", "order_items", "Prix ≤ 0", (it["price"] <= 0).sum(), len(it),
            "Prix de vente nul ou négatif", "Exclure")
    iss.add("Validité", "order_items", "Frais de livraison négatifs",
            (it["freight_value"] < 0).sum(), len(it), "Montant négatif", "Exclure")
    iss.add("Validité", "order_items", "Frais de livraison = 0",
            (it["freight_value"] == 0).sum(), len(it),
            "Livraison gratuite ou valeur non saisie", "Conserver (livraison offerte possible)")
    iss.add("Validité", "payments", "Paiement de valeur ≤ 0", (p["payment_value"] <= 0).sum(), len(p),
            "Montant payé nul", "Signaler")
    iss.add("Validité", "payments", "Mode de paiement 'not_defined'",
            (p["payment_type"] == "not_defined").sum(), len(p),
            "Mode de paiement non renseigné", "Regrouper en 'non_renseigne'")
    iss.add("Validité", "reviews", "Note d'avis hors de [1 ; 5]",
            (~r["review_score"].between(1, 5)).sum(), len(r), "Note invalide", "Exclure")
 
    items_total = it.groupby("order_id")[["price", "freight_value"]].sum().sum(axis=1)
    pay_total = p.groupby("order_id")["payment_value"].sum()
    both = pd.concat([items_total.rename("items"), pay_total.rename("paid")], axis=1, join="inner")
    gap = (both["items"] - both["paid"]).abs() > 1
    iss.add("Cohérence", "payments", "Écart > 1 entre montant payé et total des articles",
            gap.sum(), len(both),
            "Le total payé ne correspond pas aux articles + frais (POS et paiement non rapprochés)",
            "Utiliser le montant des articles pour le chiffre d'affaires ; signaler l'écart")
 
    # Reviews: same order reviewed several times
    n = int(r.duplicated(subset="order_id").sum())
    iss.add("Unicité", "reviews", "Plusieurs avis pour une même commande", n, len(r),
            "La commande a reçu plusieurs avis", "Garder l'avis le plus récent")
 
    # Product catalogue
    iss.add("Complétude", "products", "Produits sans catégorie",
            pr["product_category_name"].isna().sum(), len(pr),
            "Catégorie non renseignée dans le catalogue", "Catégorie 'non_renseignee'")
    known = set(cat["product_category_name"])
    cats = pr["product_category_name"].dropna().unique()
    no_tr = [x for x in cats if x not in known]
    iss.add("Complétude", "categories", "Catégories sans traduction",
            len(no_tr), len(cats), "Catégories absentes de la table de traduction : " + ", ".join(no_tr),
            "Ajouter la traduction manuellement")
    iss.add("Validité", "products", "Poids produit manquant ou = 0",
            (pr["product_weight_g"].isna() | (pr["product_weight_g"] == 0)).sum(), len(pr),
            "Donnée logistique manquante", "Signaler")
 
    # Spelling variants of city names
    for name, df, col in [("customers", c, "customer_city"), ("sellers", s, "seller_city")]:
        raw = df[col].nunique()
        norm = normalize_text(df[col]).nunique()
        iss.add("Cohérence", name, f"Variantes d'orthographe des villes ({col})", raw - norm, raw,
                "Même ville écrite différemment (accents, majuscules, espaces)",
                "Normaliser les noms de ville")
 
    # Geolocation duplicates
    n = int(g.duplicated(subset="geolocation_zip_code_prefix").sum())
    iss.add("Unicité", "geolocation", "Codes postaux répétés", n, len(g),
            "Plusieurs coordonnées pour un même code postal",
            "Une seule coordonnée moyenne par code postal")
 
    # ------------------------------------------------------------------
    # 6. Period coverage
    # ------------------------------------------------------------------
    progress("Couverture de la période")
    monthly = (o.assign(mois=buy.dt.to_period("M").astype(str))
                 .groupby("mois").size().rename("nb_commandes").reset_index())
    median = monthly["nb_commandes"].median()
    monthly["pct_de_la_mediane"] = (100 * monthly["nb_commandes"] / median).round(1)
    monthly["mois_incomplet"] = np.where(monthly["pct_de_la_mediane"] < 10, "oui", "non")
    n = (monthly["mois_incomplet"] == "oui").sum()
    iss.add("Couverture", "orders", "Mois avec un volume anormalement faible (<10 % de la médiane)",
            n, len(monthly),
            "Mois de démarrage ou de fin d'extraction : "
            + ", ".join(monthly.loc[monthly["mois_incomplet"] == "oui", "mois"]),
            "Limiter l'analyse à une période complète (2017-01 → 2018-08)")
 
    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    anomalies = iss.to_frame()
    anomalies = anomalies[anomalies["nb_cas"] > 0].sort_values(
        ["dimension_qualite", "pourcentage"], ascending=[True, False])
 
    progress("Écriture du rapport Excel")
    out_file = out_dir / "rapport_qualite_donnees.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as xw:
        overview.to_excel(xw, sheet_name="synthese_tables", index=False)
        missing.to_excel(xw, sheet_name="valeurs_manquantes", index=False)
        anomalies.to_excel(xw, sheet_name="anomalies", index=False)
        monthly.to_excel(xw, sheet_name="commandes_par_mois", index=False)
        for ws in xw.book.worksheets:  # readable column widths
            for col in ws.columns:
                width = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)
            ws.freeze_panes = "A2"
 
    print("\n=== Anomalies détectées ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(anomalies[["table", "controle", "nb_cas", "pourcentage"]].to_string(index=False))
    print(f"\nRapport écrit : {out_file}   (durée totale : {time.time() - T0:.1f} s)")
 
 
if __name__ == "__main__":
    main()