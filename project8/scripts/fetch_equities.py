"""Fetch aligned daily adjusted closes and save an external research snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SYMBOLS = ("SPY", "QQQ", "IWM", "DIA")


def fetch(output):
    series = []
    for symbol in SYMBOLS:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1=1262131200&period2=1780272000&interval=1d&events=history"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as stream:
            record = json.load(stream)["chart"]["result"][0]
        if record["meta"]["symbol"] != symbol or record["meta"]["currency"] != "USD":
            raise ValueError(f"Unexpected instrument: {symbol}")
        dates = (
            pd.to_datetime(record["timestamp"], unit="s", utc=True)
            .tz_convert("America/New_York")
            .date
        )
        series.append(
            pd.Series(record["indicators"]["adjclose"][0]["adjclose"], index=dates, name=symbol)
        )
    frame = pd.concat(series, axis=1).sort_index()
    if frame.isna().any().any() or not frame.index.is_unique or (frame <= 0).any().any():
        raise ValueError("Incomplete trading-date alignment")
    frame.to_csv(output, index_label="date", float_format="%.10f")
    print(
        f"{len(frame)} closes per fund; SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    fetch(p.parse_args().output)
