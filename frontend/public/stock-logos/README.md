These local assets correspond to the logos and instrument icons listed in
`frontend/lib/stock-logo-manifest.json`. The catalogue includes Wikimedia Commons
previews, brapi issuer icons, a Fintz icon, Parqet issuer icons for US equities
(shared with their corresponding B3 BDRs), and images from issuer sites,
Financial Modeling Prep, TradingView, CoinGecko, and MWM. Logos are bundled
locally so the interface does not depend on those sources at runtime. Share
classes of the same company in the local B3 list share one asset. Funds may
share their manager's logo. Reference rates, unbranded fixed-income products,
commodities, and the currency pair use instrument icons.

The JSON manifest records the source, author, and license information for each
asset. It is also the input for `python scripts/fetch_stock_logos.py`, which
downloads the images and verifies their formats. For PNGs, it checks the signature
and dimensions. For SVGs,
it rejects scripts, event handlers, external references, entities, and elements
outside a small allowlist for static graphics and metadata.

`python scripts/fetch_stock_logos.py --check` validates the bundled assets without
network requests. It fails on missing or invalid images, duplicate ticker
mappings, or listed tickers without logos. CI runs this check against
`acoes-listadas-b3.csv` and uses pytest to verify coverage of every ticker in
the local multi-asset catalog.

The logos remain trademarks of their respective issuers. The WEG logo is
licensed CC BY-SA 4.0; the manifest provides the attribution and source link.

### Lucide instrument icons

The bank deposit, currency exchange, fixed-income, reference-rate, and
savings icons are distributed under the ISC License.

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
