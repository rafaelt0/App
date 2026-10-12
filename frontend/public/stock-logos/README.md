These local assets correspond to the issuer logos listed in
`frontend/lib/stock-logo-manifest.json`. The PNGs are rasterized previews of
their original logos; Vale's standalone SVG is copied unchanged from Vale's own
investor site. The JSON manifest records the source, author, and license for
each asset. It is also the input for `python scripts/fetch_stock_logos.py`,
which rejects non-PNG responses and verifies PNG headers/dimensions. For SVGs,
it rejects scripts, event handlers, external references, entities, and elements
outside a small static drawing allowlist.

The logos remain trademarks of their respective issuers. The WEG logo is
licensed CC BY-SA 4.0; the manifest provides the attribution and source link.
