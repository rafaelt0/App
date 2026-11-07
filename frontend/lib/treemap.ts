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
