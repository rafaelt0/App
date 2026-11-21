"use client";

import { useEffect, useLayoutEffect, useState, type RefObject } from "react";

export type CursorState = {
  target: string | null;
  offsetX?: number;
  offsetY?: number;
  visible: boolean;
  pressing: boolean;
  clickSeq: number;
  badge?: string | null;
};

export const idleCursor: CursorState = {
  target: null,
  visible: false,
  pressing: false,
  clickSeq: 0,
  badge: null,
};

type Point = { x: number; y: number; flipLeft: boolean; flipUp: boolean; ready: boolean };

/**
 * Choreographed pointer overlay measured against real `[data-demo-target]` elements
 * inside the tutorial stage so the pointer tip lands accurately at every viewport size.
 */
export default function TutorialCursor({
  stageRef,
  cursor,
  layoutKey,
}: {
  stageRef: RefObject<HTMLDivElement | null>;
  cursor: CursorState;
  layoutKey: string;
}) {
  const [point, setPoint] = useState<Point>({ x: 0, y: 0, flipLeft: false, flipUp: false, ready: false });

  const measure = () => {
    const stage = stageRef.current;
    if (!stage || !cursor.target) return false;
    const element = stage.querySelector<HTMLElement>(`[data-demo-target="${cursor.target}"]`);
    if (!element) return false;
    const stageRect = stage.getBoundingClientRect();
    const targetRect = element.getBoundingClientRect();
    if (!stageRect.width || !targetRect.width) return false;
    const x = targetRect.left - stageRect.left + targetRect.width * (cursor.offsetX ?? 0.56);
    const y = targetRect.top - stageRect.top + targetRect.height * (cursor.offsetY ?? 0.58);
    setPoint({
      x,
      y,
      flipLeft: x > stageRect.width - 165,
      flipUp: y > stageRect.height - 52,
      ready: true,
    });
    return true;
  };

  useLayoutEffect(() => {
    if (!cursor.target) return;
    if (measure()) return;
    const frame = window.requestAnimationFrame(() => {
      measure();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [cursor.target, cursor.offsetX, cursor.offsetY, layoutKey]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !cursor.target) return;
    const observer = new ResizeObserver(() => {
      measure();
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, [cursor.target, cursor.offsetX, cursor.offsetY, stageRef]);

  if (!cursor.visible || !point.ready) return null;

  return (
    <div
      className={`tut-cursor${cursor.pressing ? " is-pressing" : ""}${point.flipLeft ? " is-left" : ""}${point.flipUp ? " is-up" : ""}`}
      style={{ transform: `translate3d(${point.x}px, ${point.y}px, 0)` }}
      aria-hidden="true"
    >
      {cursor.clickSeq > 0 && <span key={cursor.clickSeq} className="tut-cursor-ripple" />}
      <svg className="tut-cursor-arrow" width="22" height="24" viewBox="0 0 22 24" fill="none">
        <path
          d="M2.2 1.8L2.2 18.6L6.9 14.3L10.3 21.8L13.1 20.5L9.7 13.1L16.3 12.6L2.2 1.8Z"
          fill="#f3f5f9"
          stroke="#050607"
          strokeWidth="1.8"
          strokeLinejoin="miter"
        />
        <rect x="4.5" y="6.8" width="2.5" height="2.5" fill="#ffb224" />
      </svg>
      {cursor.badge && <span className="tut-cursor-badge">{cursor.badge}</span>}
    </div>
  );
}
