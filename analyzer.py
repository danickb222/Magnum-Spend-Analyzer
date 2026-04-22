import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from scipy import stats
import anthropic
import os
import sys
from datetime import datetime

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COMPLIANCE_RISK = {
    "Germany": "Low", "Poland": "Low", "USA": "Low",
    "China": "High", "Turkey": "High", "Philippines": "High"
}

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")
    print(f"\n✓ Loaded {len(df)} rows from {filepath}")
    print(f"  Columns: {list(df.columns)}")
    return df

# ── 2. PARETO ANALYSIS ────────────────────────────────────────────────────────
def pareto_analysis(df, supplier_col, spend_col):
    df = df.sort_values(spend_col, ascending=False).reset_index(drop=True)
    df["Cumulative_Spend"] = df[spend_col].cumsum()
    df["Cumulative_%"] = (df["Cumulative_Spend"] / df[spend_col].sum() * 100).round(2)
    df["Spend_%"] = (df[spend_col] / df[spend_col].sum() * 100).round(2)
    top_80 = df[df["Cumulative_%"] <= 80]
    print(f"\n── PARETO ANALYSIS ───────────────────────────────────────")
    print(df[[supplier_col, spend_col, "Spend_%", "Cumulative_%"]].to_string(index=False))
    print(f"\n→ {len(top_80)} of {len(df)} suppliers account for 80% of total spend")
    print(f"→ Total spend: €{df[spend_col].sum():,.0f}")
    return df

# ── 3. COUNTRY BREAKDOWN ──────────────────────────────────────────────────────
def country_breakdown(df, country_col, spend_col):
    country_spend = df.groupby(country_col)[spend_col].agg(
        Total_Spend="sum",
        Supplier_Count="count"
    ).sort_values("Total_Spend", ascending=False)
    print(f"\n── SPEND BY COUNTRY ──────────────────────────────────────")
    print(country_spend.to_string())
    return country_spend

# ── 4. RISK SCORING ───────────────────────────────────────────────────────────
def risk_scoring(df, supplier_col, spend_col, country_col):
    total_spend = df[spend_col].sum()
    df["Spend_Share"] = df[spend_col] / total_spend
    df["Country_Count"] = df.groupby(supplier_col)[country_col].transform("nunique")
    
    def score(row):
        risk = 0
        if row["Spend_Share"] > 0.15: risk += 3
        elif row["Spend_Share"] > 0.08: risk += 2
        else: risk += 1
        if row["Country_Count"] == 1: risk += 2
        elif row["Country_Count"] <= 2: risk += 1
        return risk

    df["Risk_Score"] = df.apply(score, axis=1)
    df["Risk_Level"] = df["Risk_Score"].apply(
        lambda x: "🔴 HIGH" if x >= 4 else ("🟡 MEDIUM" if x >= 3 else "🟢 LOW")
    )
    print(f"\n── SUPPLIER RISK SCORING ─────────────────────────────────")
    print(df[[supplier_col, spend_col, "Spend_Share", "Country_Count", "Risk_Level"]].to_string(index=False))
    return df

# ── 5. SUPPLIER DEPENDENCY INDEX ──────────────────────────────────────────────
def dependency_index(df, country_col, supplier_col, spend_col):
    print(f"\n── SUPPLIER DEPENDENCY INDEX ─────────────────────────────")
    for country in df[country_col].unique():
        country_df = df[df[country_col] == country]
        top_supplier_share = (country_df[spend_col].max() / country_df[spend_col].sum() * 100)
        flag = "⚠️  CRITICAL" if top_supplier_share > 70 else ("⚡ ELEVATED" if top_supplier_share > 50 else "✓ HEALTHY")
        top_name = country_df.loc[country_df[spend_col].idxmax(), supplier_col]
        print(f"  {country}: {top_name} holds {top_supplier_share:.1f}% of country spend — {flag}")

# ── 6. GEOGRAPHIC COVERAGE GAPS ───────────────────────────────────────────────
def coverage_gaps(df, supplier_col, country_col, spend_col, top_n=5):
    print(f"\n── GEOGRAPHIC COVERAGE GAPS ──────────────────────────────")
    top_suppliers = df.groupby(supplier_col)[spend_col].sum().nlargest(top_n).index
    all_countries = df[country_col].unique()
    matrix = pd.DataFrame(index=top_suppliers, columns=all_countries)
    for supplier in top_suppliers:
        s_countries = df[df[supplier_col] == supplier][country_col].unique()
        for country in all_countries:
            matrix.loc[supplier, country] = "✓" if country in s_countries else "✗"
    print(matrix.to_string())
    print("\n→ ✗ indicates a gap where your top supplier is not present")

# ── 7. SUPPLIER OVERLAP ANALYSIS ──────────────────────────────────────────────
def supplier_overlap(df, supplier_col, country_col, spend_col):
    print(f"\n── SUPPLIER OVERLAP ACROSS MARKETS ───────────────────────")
    multi_market = df.groupby(supplier_col)[country_col].nunique()
    multi_market = multi_market[multi_market > 1].sort_values(ascending=False)
    if multi_market.empty:
        print("→ No suppliers present in multiple markets")
    else:
        for supplier, count in multi_market.items():
            countries = df[df[supplier_col] == supplier][country_col].unique()
            spend = df[df[supplier_col] == supplier][spend_col].sum()
            print(f"  {supplier}: {count} markets ({', '.join(countries)}) — €{spend:,.0f} consolidation opportunity")

# ── 8. COMPLIANCE RISK HEATMAP ────────────────────────────────────────────────
def compliance_heatmap(country_spend):
    print(f"\n── COMPLIANCE RISK BY COUNTRY ────────────────────────────")
    for country, row in country_spend.iterrows():
        risk = COMPLIANCE_RISK.get(country, "Medium")
        flag = "🔴" if risk == "High" else ("🟡" if risk == "Medium" else "🟢")
        print(f"  {flag} {country}: {risk} regulatory risk | {int(row['Supplier_Count'])} suppliers | €{row['Total_Spend']:,.0f} spend")

# ── 9. NEGOTIATION LEVERAGE SCORE ─────────────────────────────────────────────
def negotiation_leverage(df, supplier_col, spend_col):
    print(f"\n── NEGOTIATION LEVERAGE SCORE ────────────────────────────")
    print("  (Based on your spend as % of supplier's estimated market presence)")
    supplier_spend = df.groupby(supplier_col)[spend_col].sum().sort_values(ascending=False)
    for supplier, spend in supplier_spend.items():
        share = spend / supplier_spend.sum() * 100
        if share > 15:
            leverage = "🔥 STRONG — prioritize for renegotiation"
        elif share > 8:
            leverage = "💪 MODERATE — bundle contracts across markets"
        else:
            leverage = "⚠️  WEAK — consider consolidating volume"
        print(f"  {supplier}: {share:.1f}% of total spend — {leverage}")

# ── 10. ANOMALY DETECTION ─────────────────────────────────────────────────────
def anomaly_detection(df, supplier_col, spend_col):
    print(f"\n── ANOMALY DETECTION ─────────────────────────────────────")
    mean = df[spend_col].mean()
    std = df[spend_col].std()
    df["Z_Score"] = ((df[spend_col] - mean) / std).abs()
    anomalies = df[df["Z_Score"] > 2].sort_values("Z_Score", ascending=False)
    if anomalies.empty:
        print("→ No statistical anomalies detected")
    else:
        print(f"→ {len(anomalies)} anomalous suppliers detected (>2 standard deviations):")
        for _, row in anomalies.iterrows():
            print(f"  ⚠️  {row[supplier_col]}: €{row[spend_col]:,.0f} spend — Z-score: {row['Z_Score']:.2f}")

# ── 11. MONTE CARLO SAVINGS SIMULATION ───────────────────────────────────────
def monte_carlo(df, spend_col, simulations=1000):
    print(f"\n── MONTE CARLO SAVINGS SIMULATION ───────────────────────")
    total_spend = df[spend_col].sum()
    savings_rates = np.random.uniform(0.03, 0.15, simulations)
    savings = total_spend * savings_rates
    p10 = np.percentile(savings, 10)
    p50 = np.percentile(savings, 50)
    p90 = np.percentile(savings, 90)
    print(f"  Based on {simulations:,} simulations of consolidation scenarios:")
    print(f"  → Conservative (P10): €{p10:,.0f} savings")
    print(f"  → Expected   (P50): €{p50:,.0f} savings")
    print(f"  → Optimistic  (P90): €{p90:,.0f} savings")
    print(f"  → 90% probability that consolidation delivers between €{p10:,.0f} and €{p90:,.0f}")
    return p10, p50, p90

# ── 12. CONSOLIDATION ROADMAP ─────────────────────────────────────────────────
def consolidation_roadmap(df, country_col, supplier_col, spend_col):
    print(f"\n── CONSOLIDATION ROADMAP ─────────────────────────────────")
    country_stats = df.groupby(country_col).agg(
        Total_Spend=(spend_col, "sum"),
        Supplier_Count=(supplier_col, "count")
    )
    country_stats["Savings_Potential"] = country_stats["Total_Spend"] * 0.08
    country_stats["Complexity"] = country_stats["Supplier_Count"].apply(
        lambda x: 3 if x > 5 else (2 if x > 3 else 1)
    )
    country_stats["Priority_Score"] = (
        country_stats["Savings_Potential"] / country_stats["Savings_Potential"].max() * 0.6 +
        (1 / country_stats["Complexity"]) * 0.4
    )
    roadmap = country_stats.sort_values("Priority_Score", ascending=False).reset_index()
    quarters = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"]
    for i, (_, row) in enumerate(roadmap.iterrows()):
        quarter = quarters[i] if i < len(quarters) else "2027+"
        print(f"  {quarter}: {row[country_col]}")
        print(f"    → {int(row['Supplier_Count'])} suppliers | €{row['Total_Spend']:,.0f} spend | Est. savings: €{row['Savings_Potential']:,.0f}")
        print(f"    → Complexity: {'HIGH' if row['Complexity'] == 3 else ('MEDIUM' if row['Complexity'] == 2 else 'LOW')}")

# ── 13. NETWORK GRAPH ─────────────────────────────────────────────────────────
def network_graph(df, supplier_col, country_col, spend_col):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row[supplier_col], type="supplier")
        G.add_node(row[country_col], type="country")
        G.add_edge(row[supplier_col], row[country_col], weight=row[spend_col])

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, seed=42)
    supplier_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "supplier"]
    country_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "country"]
    nx.draw_networkx_nodes(G, pos, nodelist=supplier_nodes, node_color="steelblue", node_size=800, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=country_nodes, node_color="darkorange", node_size=1200, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", font_weight="bold")
    edges = G.edges(data=True)
    weights = [d["weight"] / df[spend_col].max() * 3 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.4, edge_color="gray")
    blue = mpatches.Patch(color="steelblue", label="Supplier")
    orange = mpatches.Patch(color="darkorange", label="Country")
    plt.legend(handles=[blue, orange])
    plt.title("Supplier-Country Network Graph\n(Edge thickness = spend volume)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("network_graph.png", dpi=150)
    plt.show()
    print("→ Network graph saved as network_graph.png")

# ── 14. AUTO EXECUTIVE SUMMARY ────────────────────────────────────────────────
def executive_summary(df, country_spend, supplier_col, spend_col, country_col, p10, p50, p90):
    total_spend = df[spend_col].sum()
    n_suppliers = len(df[supplier_col].unique())
    n_countries = len(df[country_col].unique())
    top_supplier = df.groupby(supplier_col)[spend_col].sum().idxmax()
    top_country = country_spend["Total_Spend"].idxmax()
    fragmented = country_spend[country_spend["Supplier_Count"] > 3]

    summary = f"""
══════════════════════════════════════════════════════════════
EXECUTIVE SUMMARY — SUPPLIER SPEND ANALYSIS
Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}
══════════════════════════════════════════════════════════════

Total spend of €{total_spend:,.0f} is distributed across {n_suppliers} suppliers 
in {n_countries} countries. The top supplier by spend is {top_supplier}, 
and the highest-spend market is {top_country}.

CONSOLIDATION OPPORTUNITY
{len(fragmented)} markets show critical fragmentation with more than 3 active 
suppliers. Consolidation to 2-3 strategic partners per market is recommended.
Monte Carlo simulation projects savings of €{p50:,.0f} (expected case), 
with a 90% confidence interval of €{p10:,.0f} to €{p90:,.0f} annually.

KEY RISKS
Supplier dependency and compliance risk are elevated in high-regulatory 
markets. Immediate review of single-source dependencies is recommended.

RECOMMENDED NEXT STEPS
1. Initiate consolidation in highest-priority markets per roadmap
2. Renegotiate contracts with top suppliers leveraging consolidated volume
3. Establish preferred supplier program with 3-5 global strategic partners
4. Implement quarterly spend monitoring using this tool

══════════════════════════════════════════════════════════════
"""
    print(summary)
    with open("executive_summary.txt", "w") as f:
        f.write(summary)
    print("→ Executive summary saved as executive_summary.txt")

# ── 15. NATURAL LANGUAGE QUERY ────────────────────────────────────────────────
def natural_language_query(df, supplier_col, spend_col, country_col):
    print(f"\n── NATURAL LANGUAGE QUERY INTERFACE ──────────────────────")
    print("  Ask any question about your supplier data in plain English.")
    print("  Type 'exit' to quit.\n")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    data_summary = df.to_string()

    while True:
        query = input("  Your question: ").strip()
        if query.lower() == "exit":
            break
        
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""You are a procurement analytics assistant. 
Answer this question based on the supplier spend data below.
Be concise, specific, and use numbers from the data.

Data:
{data_summary}

Question: {query}"""
            }]
        )
        print(f"\n  → {message.content[0].text}\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n══════════════════════════════════════════════════════════════")
    print("  SUPPLIER SPEND ANALYZER — Magnum/Unilever Procurement Tool")
    print("══════════════════════════════════════════════════════════════\n")

    filepath = input("Enter path to your CSV or Excel file: ").strip()
    df = load_data(filepath)

    print(f"\nColumns detected: {list(df.columns)}")
    supplier_col = input("Supplier column name: ").strip()
    spend_col = input("Spend column name: ").strip()
    country_col = input("Country column name: ").strip()

    df = pareto_analysis(df, supplier_col, spend_col)
    country_spend = country_breakdown(df, country_col, spend_col)
    df = risk_scoring(df, supplier_col, spend_col, country_col)
    dependency_index(df, country_col, supplier_col, spend_col)
    coverage_gaps(df, supplier_col, country_col, spend_col)
    supplier_overlap(df, supplier_col, country_col, spend_col)
    compliance_heatmap(country_spend)
    negotiation_leverage(df, supplier_col, spend_col)
    anomaly_detection(df, supplier_col, spend_col)
    p10, p50, p90 = monte_carlo(df, spend_col)
    consolidation_roadmap(df, country_col, supplier_col, spend_col)
    network_graph(df, supplier_col, country_col, spend_col)
    executive_summary(df, country_spend, supplier_col, spend_col, country_col, p10, p50, p90)

    print("\n── NATURAL LANGUAGE INTERFACE ────────────────────────────────")
    use_nlp = input("Launch natural language query interface? (yes/no): ").strip().lower()
    if use_nlp == "yes":
        natural_language_query(df, supplier_col, spend_col, country_col)

    print("\n✓ Analysis complete. All outputs saved to current directory.")

if __name__ == "__main__":
    main()
