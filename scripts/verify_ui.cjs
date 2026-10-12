// Uses an external Playwright installation; no application dependency is added.
// B3LAB_PLAYWRIGHT_MODULE=/path/to/playwright node scripts/verify_ui.cjs [url] [output-directory]
// Set B3LAB_UI_ANALYSES=1 to also exercise analyses with real market providers.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require(process.env.B3LAB_PLAYWRIGHT_MODULE || "playwright");

const baseUrl = process.argv[2] || "http://localhost:3000";
const output = path.resolve(process.argv[3] || "/tmp/b3lab-ui-verification");
const routes = ["/", "/portfolio", "/simulacao", "/noticias", "/screener"];
const widths = [375, 768, 1024, 1440];

async function main() {
  await fs.mkdir(output, { recursive: true });
  const layout = await fs.readFile(path.join(__dirname, "check_layout.js"), "utf8");
  const surfaces = await fs.readFile(path.join(__dirname, "check_ui_layout.js"), "utf8");
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ reducedMotion: "reduce" });
  const page = await context.newPage();
  const errors = [];
  const reports = [];
  page.on("pageerror", error => errors.push(error.message));

  async function settled() {
    await page.locator("h1").waitFor();
    await page.waitForFunction(() => !document.querySelector(".loading-container"));
    await page.waitForLoadState("networkidle");
    await page.waitForFunction(() => !document.querySelector(".loading-container"));
    await page.evaluate(() => document.fonts.ready);
  }

  async function check(state) {
    const report = await page.evaluate(layout);
    report.state = state;
    reports.push(report);
    try {
      const surface = await page.evaluate(surfaces);
      report.failures.push(...surface.failures);
    } catch (error) {
      report.failures.push(error.message);
    }
    console.log(JSON.stringify(report));
  }

  try {
    for (const width of widths) {
      await page.setViewportSize({ width, height: 1000 });
      for (const route of routes) {
        await page.goto(new URL(route, baseUrl).href);
        await settled();
        await check("initial");
        const name = route === "/" ? "home" : route.slice(1);
        await page.screenshot({ path: path.join(output, `${name}-${width}.png`), fullPage: true });
        await page.locator("details").evaluateAll(items => items.forEach(item => { item.open = true; }));
        await check("disclosures-open");
      }
    }

    // A search selection is persisted by the real API without running market analyses.
    await page.goto(baseUrl);
    await settled();
    const picker = page.getByRole("searchbox", { name: /^Buscar ações/ });
    await picker.fill("WEGE3");
    await picker.press("ArrowDown");
    await picker.press("ArrowUp");
    await picker.press("Enter");
    await page.getByRole("button", { name: "Remover WEGE3", exact: true }).waitFor();
    await check("ticker-selected");
    await page.getByRole("button", { name: "Adicionar WEGE3 aos favoritos", exact: true }).first().click();
    await page.waitForFunction(() => document.querySelector('.ticker-chip-favorite[aria-label="Remover WEGE3 dos favoritos"]')?.getAttribute("aria-pressed") === "true");
    await page.screenshot({ path: path.join(output, "home-selected-1440.png"), fullPage: true });

    await page.setViewportSize({ width: 375, height: 1000 });
    await check("ticker-selected-mobile");
    const menu = page.getByRole("button", { name: "Menu", exact: true });
    await menu.click();
    assert.equal(await menu.getAttribute("aria-expanded"), "true");
    await page.getByRole("link", { name: "Portfólio", exact: true }).first().focus();
    await page.keyboard.press("Escape");
    assert.equal(await menu.getAttribute("aria-expanded"), "false");
    assert.equal(await menu.evaluate(element => element === document.activeElement), true);

    await page.keyboard.press("Control+k");
    await page.locator("dialog[open]").waitFor();
    await page.keyboard.press("Escape");
    await page.waitForFunction(() => !document.querySelector("dialog[open]"));
    await page.getByRole("button", { name: "Remover WEGE3", exact: true }).click();
    await page.getByRole("heading", { name: "Ou escolha uma ação líquida", exact: true }).waitFor();
    await check("ticker-removed-mobile");

    // Route CSS remains loaded after navigation; repeat without a document reload.
    await page.setViewportSize({ width: 1440, height: 1000 });
    for (const route of routes.slice(1)) {
      await page.locator(`nav a[href="${route}"]`).click();
      await page.waitForURL(new URL(route, baseUrl).href);
      await settled();
      await check("client-navigation");
    }

    if (process.env.B3LAB_UI_ANALYSES === "1") {
      async function captureLoaded(name) {
        for (const width of widths) {
          await page.setViewportSize({ width, height: 1000 });
          await check(`${name}-loaded`);
          if (width === 375 || width === 1440) {
            await page.screenshot({ path: path.join(output, `${name}-loaded-${width}.png`), fullPage: true });
          }
        }
      }

      await page.goto(baseUrl);
      await settled();
      for (const ticker of ["PETR4", "VALE3"]) {
        const search = page.getByRole("searchbox", { name: /^Buscar ações/ });
        await search.fill(ticker);
        await search.press("Enter");
        await page.getByRole("button", { name: `Remover ${ticker}`, exact: true }).waitFor();
      }
      await page.getByRole("button", { name: "Analisar", exact: true }).click();
      await page.getByRole("tablist", { name: "Seções da análise" }).waitFor({ timeout: 120000 });
      await captureLoaded("home");
      const tabs = page.getByRole("tab");
      await tabs.first().focus();
      await page.keyboard.press("End");
      assert.equal(await tabs.last().getAttribute("aria-selected"), "true");
      await page.keyboard.press("Home");
      assert.equal(await tabs.first().getAttribute("aria-selected"), "true");
      for (const tab of await tabs.all()) {
        await tab.click();
        await check(`home-tab-${await tab.textContent()}`);
      }

      await page.goto(new URL("/portfolio", baseUrl).href);
      await settled();
      await page.getByRole("button", { name: "Carregar portfólio", exact: true }).click();
      await page.locator(".pf-results").waitFor({ timeout: 120000 });
      await settled();
      await captureLoaded("portfolio");

      await page.goto(new URL("/simulacao", baseUrl).href);
      await page.locator(".sim-form").waitFor({ timeout: 120000 });
      await page.getByRole("spinbutton", { name: "Número de simulações (trajetórias)" }).fill("200");
      await page.getByRole("spinbutton", { name: "Horizonte (anos)" }).fill("1");
      await page.getByRole("button", { name: "Rodar Simulação", exact: true }).click();
      await page.locator(".sim-results").waitFor({ timeout: 120000 });
      await captureLoaded("simulation");

      await page.goto(new URL("/noticias", baseUrl).href);
      await page.getByRole("heading", { name: "Tom da amostra disponível", exact: true }).waitFor({ timeout: 120000 });
      await settled();
      await captureLoaded("news");
    }

    assert.deepEqual(errors, [], "Browser exceptions");
    assert.equal(reports.filter(report => report.failures.length).length, 0, "Layout failures; see report.json");
  } finally {
    await fs.writeFile(path.join(output, "report.json"), JSON.stringify({ baseUrl, reports, errors }, null, 2));
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
