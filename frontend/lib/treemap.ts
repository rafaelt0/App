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

export type Group<T> = { key: string; items: (T & { value: number })[] };
export type LaidGroup<T> = Rect & { key: string; header: boolean; items: (T & { value: number } & Rect)[] };

/**
 * Two-level treemap: groups are squarified by their summed values, then each group's items inside its own rect. A group
 * gets a `headerHeight` strip on top when it is tall and wide enough to hold a label above at least one row of items.
 */
export function nestedSquarify<T>(groups: readonly Group<T>[], box: Rect, headerHeight: number, gap = 1): LaidGroup<T>[] {
  const totals = groups.map(group => ({ group, value: group.items.reduce((sum, item) => sum + Math.max(0, item.value), 0) }));
  return squarify(totals, box).map(({ group, x, y, w, h }) => {
    const header = h >= headerHeight * 2.5 && w >= 48;
    const inner = { x: x + gap, y: y + (header ? headerHeight : gap), w: Math.max(0, w - 2 * gap), h: Math.max(0, h - gap - (header ? headerHeight : gap)) };
    return { key: group.key, x, y, w, h, header, items: squarify(group.items, inner) };
  });
}
