import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { FiChevronDown } from 'react-icons/fi';
import cs from './ChipSelect.module.css';

function normStr(s) {
  return String(s)
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase();
}

/**
 * Liste déroulante présentée comme un bouton chip (style CommandBar).
 * @param {{ value: string|number, label: string }[]} options
 * @param {boolean} [filterable] — saisie dans le chip pour filtrer les options (combobox)
 */
export default function ChipSelect({
  options,
  value,
  onChange,
  placeholder = 'Sélectionner',
  disabled = false,
  className = '',
  menuMinWidth = 200,
  id,
  'aria-labelledby': ariaLabelledby,
  filterable = false,
}) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef(null);
  const menuRef = useRef(null);
  const inputRef = useRef(null);
  const prevValueRef = useRef(value);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 0 });
  const [inputValue, setInputValue] = useState('');

  const hasValue = value !== '' && value != null && String(value).length > 0;

  useEffect(() => {
    if (!filterable) return;
    const sel = options.find((o) => String(o.value) === String(value));
    if (value !== prevValueRef.current) {
      prevValueRef.current = value;
      setInputValue(sel?.label ?? '');
      return;
    }
    if (sel && value !== '' && value != null) {
      setInputValue((prev) => (prev === '' || prev === placeholder ? sel.label : prev));
    }
  }, [value, options, filterable, placeholder]);

  const filteredOptions = useMemo(() => {
    if (!filterable) return options;
    const q = inputValue.trim();
    if (!q) return options;
    const nq = normStr(q);
    return options.filter((o) => normStr(o.label).includes(nq));
  }, [options, inputValue, filterable]);

  useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const reposition = useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({
      top: r.bottom + 4,
      left: r.left,
      width: Math.max(r.width, menuMinWidth),
    });
  }, [menuMinWidth]);

  useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => {
      window.removeEventListener('scroll', reposition, true);
      window.removeEventListener('resize', reposition);
    };
  }, [open, reposition]);

  const commitBlur = useCallback(() => {
    if (!filterable) return;
    const sel = options.find((o) => String(o.value) === String(value));
    const q = inputValue.trim();
    if (!q) {
      if (value) onChange('');
      setInputValue('');
      return;
    }
    const exact = options.find((o) => o.label.trim().toLowerCase() === q.toLowerCase());
    if (exact) {
      onChange(exact.value);
      setInputValue(exact.label);
      return;
    }
    if (sel) setInputValue(sel.label);
  }, [filterable, inputValue, value, onChange, options]);

  const handleInputBlur = () => {
    if (!filterable) return;
    window.setTimeout(() => {
      if (menuRef.current?.contains(document.activeElement)) return;
      if (btnRef.current?.contains(document.activeElement)) return;
      setOpen(false);
      commitBlur();
    }, 0);
  };

  const selectOption = (o) => {
    onChange(o.value);
    if (filterable) setInputValue(o.label);
    setOpen(false);
  };

  const toggleOpen = (e) => {
    e?.preventDefault();
    if (!disabled) setOpen((p) => !p);
  };

  if (filterable) {
    return (
      <div className={`${cs.wrap} ${className}`.trim()}>
        <div
          ref={btnRef}
          className={`${cs.btn} ${cs.btnCombobox} ${hasValue ? cs.btnActive : ''}`.trim()}
        >
          <input
            ref={inputRef}
            id={id}
            type="text"
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            disabled={disabled}
            className={cs.inputFilter}
            placeholder={placeholder}
            aria-expanded={open}
            aria-controls={open ? `${id || 'chip-select'}-listbox` : undefined}
            aria-autocomplete="list"
            aria-labelledby={ariaLabelledby}
            role="combobox"
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onBlur={handleInputBlur}
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                e.preventDefault();
                setOpen(false);
                inputRef.current?.blur();
              }
            }}
          />
          <button
            type="button"
            tabIndex={-1}
            disabled={disabled}
            className={cs.chevronBtn}
            aria-hidden
            onMouseDown={(e) => {
              e.preventDefault();
              toggleOpen();
            }}
          >
            <FiChevronDown size={10} className={`${cs.arrow} ${open ? cs.arrowOpen : ''}`} />
          </button>
        </div>
        {open &&
          createPortal(
            <div
              ref={menuRef}
              id={`${id || 'chip-select'}-listbox`}
              role="listbox"
              className={cs.menu}
              style={{
                position: 'fixed',
                top: pos.top,
                left: pos.left,
                width: pos.width,
                zIndex: 'var(--z-modal-popover)',
              }}
            >
              {filteredOptions.length === 0 ? (
                <div className={cs.emptyHint}>Aucun résultat</div>
              ) : (
                filteredOptions.map((o) => (
                  <button
                    key={String(o.value)}
                    type="button"
                    role="option"
                    aria-selected={String(o.value) === String(value)}
                    className={`${cs.option} ${String(o.value) === String(value) ? cs.optionActive : ''}`}
                    title={o.label}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => selectOption(o)}
                  >
                    {o.label}
                  </button>
                ))
              )}
            </div>,
            document.body
          )}
      </div>
    );
  }

  const selected = options.find((o) => String(o.value) === String(value));
  const displayLabel =
    selected?.label ??
    (value !== '' && value != null ? String(value) : placeholder);

  return (
    <div className={`${cs.wrap} ${className}`.trim()}>
      <button
        ref={btnRef}
        id={id}
        type="button"
        disabled={disabled}
        className={`${cs.btn} ${hasValue ? cs.btnActive : ''}`.trim()}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-labelledby={ariaLabelledby}
        onClick={() => !disabled && setOpen((p) => !p)}
      >
        <span className={cs.text}>{displayLabel}</span>
        <FiChevronDown size={10} className={`${cs.arrow} ${open ? cs.arrowOpen : ''}`} aria-hidden />
      </button>
      {open &&
        createPortal(
          <div
            ref={menuRef}
            role="listbox"
            className={cs.menu}
            style={{
              position: 'fixed',
              top: pos.top,
              left: pos.left,
              width: pos.width,
              zIndex: 'var(--z-modal-popover)',
            }}
          >
            {options.map((o) => (
              <button
                key={String(o.value)}
                type="button"
                role="option"
                aria-selected={String(o.value) === String(value)}
                className={`${cs.option} ${String(o.value) === String(value) ? cs.optionActive : ''}`}
                title={o.label}
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                }}
              >
                {o.label}
              </button>
            ))}
          </div>,
          document.body
        )}
    </div>
  );
}
