# Getting Started

## Install

```powershell
python -m pip install -e .
```

## Run

```powershell
python -m streamlit run app.py
```

## Main Workflow

1. Use `Market Edge` to compare model probabilities against bookmaker prices.
2. Use `Market Data` to read market-implied probabilities from sample or public
   prediction-market data.
3. Use `Football Poisson Model` to turn expected goals into football market
   probabilities.
4. Use `Model Evaluation` to test whether historical model edge became realized
   value.
5. Use `Methodology` to review formulas and trading workflow.

![Market edge](../assets/sports-edge-market-edge.png)

## Run Tests

```powershell
python -m unittest discover
```
