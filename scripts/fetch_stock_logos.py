#!/usr/bin/env python3
"""Download and validate the Commons issuer logos listed in the frontend manifest."""

from __future__ import annotations

import json
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
SAFE_SVG_ELEMENTS = {"svg", "defs", "clipPath", "rect", "g", "path"}


def main() -> int:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    failed = False

    for entry in entries:
        source_url = entry.get("sourceAssetUrl")
        if source_url is None:
            source_name = urllib.parse.quote(entry["commonsFile"], safe="")
            source_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{source_name}?width=400"
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "B3Lab local stock-logo asset fetcher/1.0"},
        )
        try:
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
                for element in root.iter():
                    tag = element.tag.rsplit("}", 1)[-1]
                    if tag not in SAFE_SVG_ELEMENTS:
                        raise ValueError(f"unsafe SVG element: <{tag}>")
                    for name, value in element.attrib.items():
                        attribute = name.rsplit("}", 1)[-1].lower()
                        if attribute.startswith("on"):
                            raise ValueError(f"unsafe SVG event attribute: {name}")
                        if attribute in {"href", "xlink:href"}:
                            raise ValueError(f"external SVG reference is not allowed: {name}")
                        if "url(" in value and value != "url(#clip-path)":
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

            destination = OUTPUT / entry["asset"]
            destination.write_bytes(payload)
            print(f"{entry['asset']}: {dimensions}, {len(payload)} bytes")
        except (OSError, ValueError, ET.ParseError, urllib.error.URLError) as error:
            failed = True
            print(f"{entry['asset']}: failed: {error}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
