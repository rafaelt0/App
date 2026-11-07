export type Rect = { x: number; y: number; w: number; h: number };

/** Squarified treemap: lays items with positive values into `box`, largest first, keeping tiles close to square. */
export function squarify<T extends { value: number }>(items: readonly T[], box: Rect): (T & Rect)[] {
  const sorted = [...items].filter(item => item.value > 0).sort((a, b) => b.value - a.value);
  const total = sorted.reduce((sum, item) => sum + item.value, 0);
  if (!total) return [];
  const scale = box.w * box.h / total;
  let rest = sorted.map(item => ({ item, area: item.value * scale }));
  let { x, y, w, h } = box;
  const placed: (T & Rect)[] = [];
  const worst = (row: { area: number }[], side: number) => {
    const sum = row.reduce((total, entry) => total + entry.area, 0);
    const largest = Math.max(...row.map(entry => entry.area));
    const smallest = Math.min(...row.map(entry => entry.area));
    return Math.max(side * side * largest / (sum * sum), sum * sum / (side * side * smallest));
  };
  while (rest.length) {
    const side = Math.min(w, h);
    let count = 1;
    while (count < rest.length && worst(rest.slice(0, count + 1), side) <= worst(rest.slice(0, count), side)) count++;
    const row = rest.slice(0, count);
    const sum = row.reduce((total, entry) => total + entry.area, 0);
    if (w >= h) {
      const width = sum / h;
      let top = y;
      for (const entry of row) { const height = entry.area / width; placed.push({ ...entry.item, x, y: top, w: width, h: height }); top += height; }
      x += width; w -= width;
    } else {
      const height = sum / w;
      let left = x;
      for (const entry of row) { const width = entry.area / height; placed.push({ ...entry.item, x: left, y, w: width, h: height }); left += width; }
      y += height; h -= height;
    }
    rest = rest.slice(count);
  }
  return placed;
}

/**
 * Wraps `text` into at most `maxLines` lines of `width` px without breaking inside a word. When the words run out of
 * room, the last line ends with an ellipsis; when not even three characters fit, the label is empty and the caller hides it.
 */
export function fitLabel(text: string, width: number, maxLines: number, measure: (value: string) => number): string[] {
  if (maxLines < 1 || width <= 0) return [];
  const fits = (value: string) => measure(value) <= width;
  const ellipsize = (value: string) => {
    let cut = value;
    while (cut.length > 3 && !fits(`${cut.replace(/[\s,.;:–-]+$/, "")}…`)) cut = cut.slice(0, -1);
    // A stub of one or two letters before the ellipsis reads as noise; end on the previous whole word instead.
    const space = cut.lastIndexOf(" ");
    if (space > 0 && cut.length < value.length && cut.length - space - 1 < 4) cut = cut.slice(0, space);
    cut = cut.replace(/\s+(e|de|do|da|dos|das)$/, "");
    const shown = `${cut.replace(/[\s,.;:–-]+$/, "")}…`;
    return cut.length > 3 || fits(shown) ? shown : null;
  };
  // Abbreviations such as "Serv.Méd.Hospit." may wrap after each period.
  const words = text.replace(/\.(?=\S)/g, ". ").split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (fits(candidate)) { line = candidate; continue; }
    if (line && lines.length + 1 < maxLines && fits(word)) { lines.push(line); line = word; continue; }
    const last = line && lines.length + 1 < maxLines ? (lines.push(line), word) : candidate;
    const closed = ellipsize(last);
    if (closed) lines.push(closed);
    else if (lines.length) lines[lines.length - 1] = ellipsize(`${lines[lines.length - 1]} ${word}`) ?? lines[lines.length - 1];
    return lines;
  }
  if (line) lines.push(line);
  return lines;
}
