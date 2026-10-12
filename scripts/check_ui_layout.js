// Run in the browser console on any B3Lab page after content has loaded.
(() => {
  const visible = element => element.getClientRects().length > 0;
  const failures = [];
  const viewport = document.documentElement.clientWidth;
  if (document.documentElement.scrollWidth > viewport + 1) {
    failures.push("The document scrolls horizontally.");
  }
  const sections = [...document.querySelectorAll(
    ".task-panel, .pf-weights, .sim-prereq, .news-task, .scr-context"
  )].filter(visible);
  for (const section of sections) {
    const style = getComputedStyle(section);
    if (style.backgroundColor !== "rgba(0, 0, 0, 0)") {
      failures.push(`${section.className} has an enclosing background.`);
    }
  }
  // Tiles may sit on a surface, but elevation comes from hairlines: no drop shadows or glow.
  const lengths = shadow => (shadow.match(/-?[\d.]+px/g) ?? []).map(parseFloat);
  for (const element of document.querySelectorAll("main *")) {
    const shadow = getComputedStyle(element).boxShadow;
    if (shadow === "none") continue;
    for (const layer of shadow.split(/,(?![^(]*\))/)) {
      if (!layer.includes("inset") && (lengths(layer)[2] ?? 0) > 0) {
        failures.push(`${element.className || element.tagName} has a blurred shadow.`);
        break;
      }
    }
  }
  const primaryActions = [...document.querySelectorAll("button.primary, a.primary")].filter(visible);
  for (const action of primaryActions) {
    if (action.getBoundingClientRect().height < 44) {
      failures.push(`${action.textContent.trim()} has a target shorter than 44px.`);
    }
  }
  const titleCount = document.querySelectorAll("h1").length;
  if (titleCount !== 1) failures.push(`Expected one page title, found ${titleCount}.`);
  const result = { page: location.pathname, viewport, sections: sections.length, failures };
  console.table(result);
  if (failures.length) throw new Error(failures.join("\n"));
  return result;
})();
