"use client";

import { useId, useLayoutEffect, useRef, type ChangeEvent, type ClipboardEvent, type KeyboardEvent } from "react";

const MAX_CENTS = 999_999_999_999_999;
const reais = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
const additions = [10, 100, 1000];

/** Currency text always shows the two cent digits, even while the user is typing. */
function formatCents(cents: number): string {
  const value = Math.trunc(cents);
  const sign = value < 0 ? "−" : "";
  const absolute = Math.abs(value);
  return `${sign}${reais.format(Math.floor(absolute / 100))},${String(absolute % 100).padStart(2, "0")}`;
}

function formatValue(value: number): string {
  return Number.isFinite(value) ? formatCents(Math.round(value * 100)) : "";
}

/** Cents are the source of truth, so a typed digit is always one more cent. */
function centsFromDigits(digits: string): number | null {
  const cents = Number(digits);
  return cents <= MAX_CENTS ? cents : null;
}

function centsFromValue(value: number): number {
  return Number.isFinite(value) ? Math.round(value * 100) : 0;
}

function countDigits(text: string): number {
  return text.replace(/\D/g, "").length;
}

/** Places the caret with the same number of digits to its right, which survives grouping. */
function caretPosition(text: string, digitsAfter: number): number {
  if (digitsAfter <= 0) return text.length;
  let seen = 0;
  for (let index = text.length - 1; index >= 0; index -= 1) {
    const code = text.charCodeAt(index);
    if (code >= 48 && code <= 57) {
      seen += 1;
      if (seen === digitsAfter) return index;
    }
  }
  return 0;
}

/** Reads pasted "R$ 1.234,56", "1,234.56" or "1.234" as a monetary amount, not as a digit stream. */
function structuredCents(text: string): number | null {
  const amount = text.replace(/^R\$\s*/i, "");
  const brazilian = /^(\d{1,3}(?:\.\d{3})+|\d*)(?:,(\d{0,2}))?$/;
  const decimalPoint = /^(\d{1,3}(?:,\d{3})+|\d*)\.(\d{0,2})$/;
  const parts = brazilian.exec(amount) ?? decimalPoint.exec(amount);
  if (!parts || !/\d/.test(amount)) return null;
  const [, reais, centavos = ""] = parts;
  return centsFromDigits(reais.replace(/\D/g, "") + centavos.padEnd(2, "0"));
}

type CapitalInputProps = {
  label: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  max?: number;
  help?: string;
  error?: string;
  onEditingChange?: (editing: boolean) => void;
};

/**
 * Controlled Brazilian-real input with a right-to-left cents mask.
 * Typing 1, 2, 3, 4, 5, 6 produces 0,01; 0,12; 1,23; 12,34; 123,45; 1.234,56.
 * A blank field reports NaN; the owning page applies business limits such as the R$ 100 minimum.
 */
export function CapitalInput({ label, value, onChange, disabled = false, max, help, error, onEditingChange }: CapitalInputProps) {
  const inputId = useId();
  const helpId = `${inputId}-help`;
  const errorId = `${inputId}-error`;
  const inputRef = useRef<HTMLInputElement>(null);
  const pendingCaret = useRef<number | null>(null);
  const text = formatValue(value);
  const describedBy = [help ? helpId : null, error ? errorId : null].filter(Boolean).join(" ") || undefined;
  const baseCents = centsFromValue(value);
  const maxCents = typeof max === "number" && Number.isFinite(max) ? Math.round(max * 100) : null;

  useLayoutEffect(() => {
    const digitsAfter = pendingCaret.current;
    const input = inputRef.current;
    if (digitsAfter == null || !input) return;
    const position = caretPosition(input.value, digitsAfter);
    input.setSelectionRange(position, position);
    pendingCaret.current = null;
  }, [text]);

  function commitCents(cents: number | null, digitsAfter = 0) {
    const formatted = cents == null ? "" : formatCents(cents);
    const input = inputRef.current;
    if (input && input === input.ownerDocument.activeElement) {
      if (formatted === text) {
        input.value = formatted;
        const position = caretPosition(formatted, digitsAfter);
        input.setSelectionRange(position, position);
      } else {
        pendingCaret.current = digitsAfter;
      }
    }
    onChange(cents == null ? Number.NaN : cents / 100);
  }

  function removeDigit(source: string, index: number) {
    const raw = source.slice(0, index) + source.slice(index + 1);
    const digits = raw.replace(/\D/g, "");
    const digitsAfter = countDigits(raw.slice(index));
    if (!digits) {
      commitCents(null, digitsAfter);
      return;
    }
    const cents = centsFromDigits(digits);
    if (cents != null) commitCents(cents, digitsAfter);
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const input = event.currentTarget;
    const raw = input.value;
    const caret = input.selectionStart ?? raw.length;
    const digitsAfter = countDigits(raw.slice(caret));
    const digits = raw.replace(/\D/g, "");
    if (!digits) {
      commitCents(null, digitsAfter);
      return;
    }
    const cents = centsFromDigits(digits);
    if (cents == null) {
      input.value = text;
      const position = caretPosition(text, digitsAfter);
      input.setSelectionRange(position, position);
      return;
    }
    commitCents(cents, digitsAfter);
  }

  function handlePaste(event: ClipboardEvent<HTMLInputElement>) {
    const pasted = event.clipboardData.getData("text").trim();
    if (/^\d+$/.test(pasted)) return;
    event.preventDefault();
    const cents = structuredCents(pasted);
    if (cents != null) commitCents(cents, 0);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (disabled || event.altKey || event.ctrlKey || event.metaKey) return;
    const input = event.currentTarget;
    const start = input.selectionStart ?? 0;
    const end = input.selectionEnd ?? 0;
    if (start !== end) return;
    if (event.key === "Backspace") {
      if (start === 0) return;
      if (input.value[start - 1] === "." || input.value[start - 1] === ",") {
        event.preventDefault();
        let index = start - 2;
        while (index >= 0 && !/\d/.test(input.value[index])) index -= 1;
        if (index >= 0) removeDigit(input.value, index);
        return;
      }
      if (start === input.value.length && input.value === formatCents(0)) {
        event.preventDefault();
        commitCents(null, 0);
      }
      return;
    }
    if (event.key !== "Delete" || start >= input.value.length) return;
    if (input.value[start] === "." || input.value[start] === ",") {
      event.preventDefault();
      let index = start + 1;
      while (index < input.value.length && !/\d/.test(input.value[index])) index += 1;
      if (index < input.value.length) removeDigit(input.value, index);
      return;
    }
    if (start === 0 && input.value === formatCents(0)) {
      event.preventDefault();
      commitCents(null, 0);
    }
  }

  function addAmount(amount: number) {
    commitCents(baseCents + amount * 100);
  }

  return <div className="capital-input">
    <label htmlFor={inputId}>{label}<span className="visually-hidden"> em reais</span></label>
    <div className="capital-input-control">
      <span className="capital-input-prefix" aria-hidden="true">R$</span>
      <input
        ref={inputRef}
        id={inputId}
        type="text"
        inputMode="numeric"
        autoComplete="off"
        autoCapitalize="off"
        autoCorrect="off"
        spellCheck={false}
        value={text}
        placeholder="0,00"
        disabled={disabled}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy}
        onFocus={event => {
          event.currentTarget.select();
          onEditingChange?.(true);
        }}
        onBlur={() => {
          pendingCaret.current = null;
          onEditingChange?.(false);
        }}
        onChange={handleChange}
        onPaste={handlePaste}
        onKeyDown={handleKeyDown}
      />
    </div>
    {help && <span className="param-hint" id={helpId}>{help}</span>}
    {error && <span className="field-error" id={errorId}>{error}</span>}
    <div className="capital-input-additions" role="group" aria-label={`Adicionar valores a ${label}`}>
      {additions.map(amount => {
        const exceeded = baseCents + amount * 100 > (maxCents ?? MAX_CENTS);
        return <button
          key={amount}
          type="button"
          disabled={disabled || exceeded}
          aria-label={`Adicionar R$ ${reais.format(amount)}`}
          onClick={() => addAmount(amount)}
        >+R$ {reais.format(amount)}</button>;
      })}
    </div>
  </div>;
}
