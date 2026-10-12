#!/usr/bin/env python3
"""Download issuer logos, or verify the bundled catalogue with --check."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frontend/lib/stock-logo-manifest.json"
OUTPUT = ROOT / "frontend/public/stock-logos"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SAFE_SVG_ELEMENTS = {
    "svg", "defs", "clipPath", "rect", "g", "path", "circle", "ellipse",
    "linearGradient", "stop", "mask",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate local assets and ticker coverage without downloading.")
    args = parser.parse_args()
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    failed = False
    covered = set()

    for entry in entries:
        for ticker in entry["tickers"]:
            if ticker in covered:
                failed = True
                print(f"Duplicate logo mapping: {ticker}", file=sys.stderr)
            covered.add(ticker)
        source_url = entry.get("sourceAssetUrl")
        if source_url is None:
            source_name = urllib.parse.quote(entry["commonsFile"], safe="")
            source_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{source_name}?width=400"
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "B3Lab local stock-logo asset fetcher/1.0"},
        )
        try:
            if args.check:
                payload = (OUTPUT / entry["asset"]).read_bytes()
                content_type = "image/svg+xml" if entry["asset"].endswith(".svg") else "image/png"
            else:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
                    content_type = response.headers.get_content_type()
            if entry["asset"].endswith(".svg"):
                if content_type != "image/svg+xml" or len(payload) > 250_000:
                    raise ValueError(f"expected a small SVG response, got {content_type}")
                if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
                    raise ValueError("SVG declarations and entities are not allowed")
                root = ET.fromstring(payload)
                if root.tag.rsplit("}", 1)[-1] != "svg":
                    raise ValueError("SVG document root is not <svg>")
                local_ids = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
                for element in root.iter():
                    tag = element.tag.rsplit("}", 1)[-1]
                    if tag not in SAFE_SVG_ELEMENTS:
                        raise ValueError(f"unsafe SVG element: <{tag}>")
                    for name, value in element.attrib.items():
                        attribute = name.rsplit("}", 1)[-1].lower()
                        if attribute.startswith("on"):
                            raise ValueError(f"unsafe SVG event attribute: {name}")
                        if attribute == "href" and (not value.startswith("#") or value[1:] not in local_ids):
                            raise ValueError(f"external SVG reference is not allowed: {name}")
                        if "url(" in value:
                            reference = re.fullmatch(r"url\(#([\w.-]+)\)", value)
                            if reference is None or reference[1] not in local_ids:
                                raise ValueError(f"external SVG URL is not allowed: {value}")
                dimensions = "validated self-contained SVG"
            else:
                if content_type != "image/png" or not payload.startswith(PNG_SIGNATURE):
                    raise ValueError(f"expected PNG response, got {content_type}")
                if len(payload) < 33:
                    raise ValueError("PNG is too small to contain an image header")

                width = int.from_bytes(payload[16:20], "big")
                height = int.from_bytes(payload[20:24], "big")
                if width < 1 or height < 1:
                    raise ValueError(f"invalid PNG dimensions: {width}x{height}")
                dimensions = f"{width}x{height}"

            if not args.check:
                destination = OUTPUT / entry["asset"]
                destination.write_bytes(payload)
                print(f"{entry['asset']}: {dimensions}, {len(payload)} bytes")
        except (OSError, ValueError, ET.ParseError, urllib.error.URLError) as error:
            failed = True
            print(f"{entry['asset']}: failed: {error}", file=sys.stderr)

    if args.check:
        with (ROOT / "acoes-listadas-b3.csv").open(encoding="utf-8") as source:
            universe = {row["Ticker"] for row in csv.DictReader(source)}
        missing = sorted(universe - covered)
        if missing:
            failed = True
            print(f"Tickers without logos: {', '.join(missing)}", file=sys.stderr)
        print(f"Validated {len(entries)} issuer assets; {len(universe & covered)}/{len(universe)} listed tickers covered.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
