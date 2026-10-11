// Evaluate this file in the browser console or T3 preview_evaluate on a loaded page.
// Repeat at 375, 768, 1024, and 1440px on each of the five routes.
// An empty failures array means every check passed.
(() => {
  const failures = [];
  const check = (condition, message) => { if (!condition) failures.push(message); };
  const visible = element => {
    const closed = element.closest("details:not([open])");
    return element.getClientRects().length > 0 && (!closed || closed.querySelector(":scope > summary")?.contains(element));
  };
  const titles = [...document.querySelectorAll("h1")].filter(visible);
  const title = titles[0];
  check(titles.length === 1, "The page must have one visible h1.");
  check(document.documentElement.scrollWidth <= innerWidth, "The page overflows horizontally.");
  check(!document.querySelector(".loading-container"), "Wait for the page to finish loading.");
  if (title) {
    const titleSize = parseFloat(getComputedStyle(title).fontSize);
    check(titleSize >= 26, "The page title must be at least 26px.");
    for (const heading of document.querySelectorAll("h2")) {
      if (visible(heading)) check(parseFloat(getComputedStyle(heading).fontSize) < titleSize, `Section heading competes with the title: ${heading.textContent.trim()}`);
    }
  }
  const icon = document.querySelector(".page-hero-icon");
  check(icon && visible(icon) && icon.getBoundingClientRect().width >= 40, "The page icon must remain visible.");
  const controls = [...document.querySelectorAll('main button, main .button-link, main input:not([type="checkbox"]):not([type="radio"]), main select, main summary')].filter(visible);
  for (const control of controls) {
    check(control.getBoundingClientRect().height >= 44, `Control is shorter than 44px: ${control.getAttribute("aria-label") || control.textContent.trim() || control.tagName}`);
  }
  for (const metric of document.querySelectorAll(".mcard")) {
    if (!visible(metric)) continue;
    const label = metric.querySelector(".mcard-label");
    const value = metric.querySelector(".mcard-value");
    check(label && value && parseFloat(getComputedStyle(value).fontSize) > parseFloat(getComputedStyle(label).fontSize), "Metric values must be larger than their labels.");
  }
  const nav = document.querySelector(".sidebar");
  const menu = document.querySelector(".mobile-header button");
  if (innerWidth < 1024) {
    check(menu && visible(menu), "Mobile navigation needs a visible Menu control.");
    check(!nav || !visible(nav) || menu?.getAttribute("aria-expanded") === "true", "Mobile navigation is open without an expanded Menu control.");
  } else {
    check(nav && visible(nav), "Desktop navigation must remain visible.");
  }
  const report = { route: location.pathname, width: innerWidth, controls: controls.length, failures };
  return report;
})()
