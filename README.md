# Supplier Spend Analyzer

Built off the back of a real global procurement consulting engagement 
analyzing tens of millions in contingent workforce spend across 60+ 
countries and hundreds of suppliers.

This tool automates the supplier concentration analysis that was done 
manually in Excel, producing 15 analytical modules in seconds.

## What it does

- Pareto analysis of supplier spend concentration
- Country-level spend breakdown
- Supplier risk scoring with traffic light system
- Supplier dependency index per market
- Geographic coverage gap analysis
- Supplier overlap across markets
- Compliance risk heatmap by country
- Negotiation leverage scoring
- Anomaly detection
- Monte Carlo savings simulation
- Consolidation roadmap with quarterly sequencing
- Network graph visualization of supplier-country relationships
- Auto-generated executive summary
- Natural language query interface powered by Claude AI

## How to use

1. Install dependencies: pip3 install pandas openpyxl matplotlib networkx scipy anthropic
2. Set your API key: export ANTHROPIC_API_KEY="your-key"
3. Run: python3 analyzer.py
4. Follow the prompts — works with any CSV or Excel file

## Built with
Python, pandas, matplotlib, networkx, scipy, Anthropic Claude API
