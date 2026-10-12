These local assets correspond to the issuer logos listed in
`frontend/lib/stock-logo-manifest.json`. The catalogue includes Wikimedia Commons
previews, brapi issuer icons, a Fintz icon, and official assets from the Vale,
PetzCobasi, and Grupo Toky investor sites. Logos are bundled locally so the
interface does not depend on those sources at runtime. Share classes of the same
company in the local B3 list share one asset.

The JSON manifest records the source, author, and license information for each
asset. It is also the input for `python scripts/fetch_stock_logos.py`, which
downloads the images and verifies their formats. For PNGs, it checks the signature
and dimensions. For SVGs,
it rejects scripts, event handlers, external references, entities, and elements
outside a small static drawing allowlist.

`python scripts/fetch_stock_logos.py --check` validates the bundled assets without
network requests. It fails on missing or invalid images, duplicate ticker
mappings, or listed tickers without logos. CI runs this check against
`acoes-listadas-b3.csv`.

The logos remain trademarks of their respective issuers. The WEG logo is
licensed CC BY-SA 4.0; the manifest provides the attribution and source link.
