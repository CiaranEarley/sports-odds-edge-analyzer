# Deployment

This guide publishes Sports Odds Edge Analyzer as a hosted Streamlit app.

## Streamlit Cloud Settings

Create the app from the GitHub repository:

```text
Repository: CiaranEarley/sports-odds-edge-analyzer
Branch: main
Main file path: app.py
Dependency file: requirements.txt
```

Choose the GitHub app flow, not a template. A template creates a new starter
project; this app already exists as a complete repository.

## Secrets

No Streamlit secrets are required for the default deployment.

The app uses:

- manual market inputs;
- bundled sample market data;
- user-uploaded CSVs for backtesting and model evaluation;
- optional read-only public Polymarket data fetched only after a button click.

No API keys are committed to the repository.

## Publish Checklist

1. Confirm `main` is pushed to GitHub.
2. In Streamlit Community Cloud, choose `Create app`.
3. Select the option for an existing app/repository.
4. Enter `CiaranEarley/sports-odds-edge-analyzer`.
5. Set branch to `main`.
6. Set main file path to `app.py`.
7. Deploy.
8. Open the app and test each tab with the bundled sample data.

## Operational Notes

The app does not need a database. Uploaded CSVs live only in the active user
session. The public Polymarket fetch is cached briefly by Streamlit and falls
back to sample data if the endpoint is unavailable.
