import argparse
import csv
import io
import json
import math
from urllib.parse import urlsplit

import requests


def main():
    parser = argparse.ArgumentParser(description="Exercita análises reais e os cinco arquivos CSV do B3Lab.")
    parser.add_argument("--url", default="http://localhost:3000")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    origin = urlsplit(base)
    headers = {"Origin": f"{origin.scheme}://{origin.netloc}", "X-B3Lab-Request": "1"}
    client = requests.Session()

    def request(path, method="GET", body=None):
        response = client.request(method, base + path, headers=headers, json=body, timeout=300)
        response.raise_for_status()
        return response

    def exported(path):
        response = request(path)
        assert response.content.startswith(b"\xef\xbb\xbf"), "CSV must retain UTF-8 BOM"
        assert "attachment; filename=" in response.headers["Content-Disposition"]
        return list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))

    context = request("/api/session").json()
    context = request("/api/session/selection", "PUT", {"tickers": ["PETR4", "VALE3"], "revision": context["revision"]}).json()
    portfolio = request("/api/portfolio/analyze", "POST", {"tickers": context["selection"], "revision": context["revision"], "strategy": "min_volatility", "capital": 10000, "lookback_days": 730, "manual_shares": {}, "gamma": 1}).json()
    assert portfolio["sample"]["count"] >= 30
    allocation = exported(f"/api/portfolio/{portfolio['id']}/allocation.csv")
    weights = {row["ticker"]: row["weight"] for row in portfolio["weights"]}
    assert set(weights) == {"PETR4", "VALE3"}
    assert math.isclose(sum(weights.values()), 1, abs_tol=1e-4)
    ticker_column = next(key for key in allocation[0] if key != "Peso (%)")
    for row in allocation:
        assert math.isclose(float(row["Peso (%)"]) / 100, weights[row[ticker_column]], abs_tol=5.1e-5)

    simulation = request("/api/simulation/run", "POST", {"revision": context["revision"], "portfolio_id": portfolio["id"], "calibration": "portfolio", "simulations": 2000, "initial_capital": 10000, "years": 1}).json()
    percentiles = exported(f"/api/simulation/{simulation['id']}/percentiles.csv")
    assert len(percentiles) == 253
    for key in ("P5", "P25", "P50", "P75", "P95"):
        assert float(percentiles[0][key]) == 10000
    assert math.isclose(float(percentiles[-1]["P5"]), simulation["stats"]["p5_terminal"], abs_tol=1e-8)
    summary = {row["Métrica"]: row["Valor"] for row in exported(f"/api/simulation/{simulation['id']}/summary.csv")}
    mean = float(summary["Valor esperado final (R$)"])
    assert math.isclose(mean, simulation["stats"]["mean_terminal"], abs_tol=.0051)
    probability = float(summary["Probabilidade de ganho (%)"]) / 100
    assert math.isclose(probability, simulation["stats"]["gain_probability"], abs_tol=.000051)

    screening = request("/api/screener/query", "POST", {"criteria": {"liq2m_min": 1000000}, "sort": {"field": "liq2m", "direction": "desc"}}).json()
    screened_csv = exported(f"/api/screener/{screening['id']}/results.csv")
    assert len(screened_csv) == screening["matched_count"]
    by_ticker = {row["Papel"]: row for row in screened_csv}
    for row in screening["rows"]:
        assert row["liq2m"] >= 1000000
        if row["dy"] is not None:
            assert math.isclose(float(by_ticker[row["ticker"]]["Div. Yield (%)"]), row["dy"] * 100, abs_tol=1e-8)
    liquidities = [row["liq2m"] for row in screening["rows"]]
    assert liquidities == sorted(liquidities, reverse=True)

    other = requests.Session()
    other.get(base + "/api/session", timeout=30).raise_for_status()
    assert other.get(base + f"/api/portfolio/{portfolio['id']}/allocation.csv", timeout=30).status_code == 404
    context = request("/api/session/selection", "PUT", {"tickers": ["WEGE3"], "revision": context["revision"]}).json()
    assert client.get(base + f"/api/simulation/{simulation['id']}/percentiles.csv", timeout=30).status_code == 404
    home = request("/api/home/analyze", "POST", {"tickers": context["selection"], "revision": context["revision"]}).json()
    fundamentals = exported(f"/api/home/{home['id']}/fundamentals.csv")
    raw_pl = float(fundamentals[0]["PL"])
    assert math.isclose(raw_pl / 100, home["companies"][0]["indicators"]["pl"], abs_tol=1e-8)
    print(json.dumps({"status": "PASS", "portfolio_sample": portfolio["sample"], "weights": weights, "normal_p5": simulation["stats"]["p5_terminal"], "bootstrap_p5": simulation["comparison"][1]["p5"], "screener_matches": screening["matched_count"], "exports": 5, "foreign_and_stale_exports": "rejected", "home_ticker": home["companies"][0]["ticker"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
