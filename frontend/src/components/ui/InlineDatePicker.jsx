import React, {
  useState, useRef, useEffect, useLayoutEffect, useCallback, forwardRef, useImperativeHandle,
} from 'react';
import { createPortal } from 'react-dom';
import { FiChevronLeft, FiChevronRight, FiCalendar } from 'react-icons/fi';
import dp from './InlineDatePicker.module.css';

const DAYS = ['Lu', 'Ma', 'Me', 'Je', 'Ve', 'Sa', 'Di'];
const MONTHS = [
  'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
  'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre',
];

const pad = (n) => String(n).padStart(2, '0');

function parseDate(str) {
  if (!str) return null;
  const [y, m, d] = str.split('-').map(Number);
  return new Date(y, m - 1, d);
}

function formatISO(date) {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function isoToDisplay(str) {
  if (!str) return '';
  const d = parseDate(str);
  if (!d) return '';
  return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`;
}

function daysInMonth(year, month) {
  return new Date(year, month + 1, 0).getDate();
}

function startDayOfWeek(year, month) {
  const day = new Date(year, month, 1).getDay();
  return day === 0 ? 6 : day - 1;
}

function applyDateMask(digits) {
  let out = '';
  for (let i = 0; i < digits.length && i < 8; i++) {
    if (i === 2 || i === 4) out += '.';
    let d = parseInt(digits[i], 10);
    if (i === 0 && d > 3) d = 3;
    if (i === 1 && parseInt(digits[0], 10) === 3 && d > 1) d = 1;
    if (i === 2 && d > 1) d = 1;
    if (i === 3 && parseInt(digits[2], 10) === 1 && d > 2) d = 2;
    out += d;
  }
  return out;
}

function displayToISO(display) {
  const parts = display.split('.');
  if (parts.length !== 3) return null;
  const [dd, mm, yyyy] = parts;
  if (dd.length !== 2 || mm.length !== 2 || yyyy.length !== 4) return null;
  const day = parseInt(dd, 10);
  const month = parseInt(mm, 10);
  const year = parseInt(yyyy, 10);
  if (month < 1 || month > 12 || day < 1) return null;
  const maxDay = daysInMonth(year, month - 1);
  if (day > maxDay) return null;
  return `${yyyy}-${mm}-${dd}`;
}

function smartComplete(digits) {
  const now = new Date();
  const todayMonth = now.getMonth();
  const todayYear = now.getFullYear();

  if (digits.length >= 1 && digits.length <= 2) {
    const day = digits.length === 1 ? parseInt(digits, 10) * 10 || parseInt(digits, 10) : parseInt(digits, 10);
    if (day < 1 || day > 31) return null;
    let m = todayMonth, y = todayYear;
    for (let t = 0; t < 13; t++) {
      const maxD = new Date(y, m + 1, 0).getDate();
      if (day <= maxD && new Date(y, m, day, 23, 59) > now) {
        return `${pad(day)}.${pad(m + 1)}.${y}`;
      }
      m++; if (m > 11) { m = 0; y++; }
    }
    return null;
  }
  if (digits.length >= 3 && digits.length <= 4) {
    const dd = parseInt(digits.slice(0, 2), 10);
    const mmPart = digits.slice(2);
    const month = mmPart.length === 1 ? parseInt(mmPart, 10) * 10 : parseInt(mmPart, 10);
    if (dd < 1 || dd > 31 || month < 1 || month > 12) return null;
    let y = todayYear;
    const maxD = new Date(y, month, 0).getDate();
    if (dd > maxD) return null;
    if (new Date(y, month - 1, dd, 23, 59) <= now) y++;
    return `${pad(dd)}.${pad(month)}.${y}`;
  }
  if (digits.length >= 5 && digits.length <= 7) {
    const dd = parseInt(digits.slice(0, 2), 10);
    const mm = parseInt(digits.slice(2, 4), 10);
    const yyyyStr = digits.slice(4).padEnd(4, '0');
    const yyyy = parseInt(yyyyStr, 10);
    if (dd < 1 || dd > 31 || mm < 1 || mm > 12 || yyyy < 2025) return null;
    const maxD = new Date(yyyy, mm, 0).getDate();
    if (dd > maxD) return null;
    return `${pad(dd)}.${pad(mm)}.${yyyy}`;
  }
  return null;
}

const InlineDatePicker = forwardRef(function InlineDatePicker({
  value,
  onChange,
  placeholder: _placeholder,
  className = '',
  inputClassName = '',
  invalid = false,
  inputId,
  ariaLabel,
  title,
}, ref) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);
  const popoverRef = useRef(null);
  const inputRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const [masked, setMasked] = useState(() => isoToDisplay(value));
  const [inputError, setInputError] = useState(false);
  const pendingSel = useRef(null);
  const isDeleting = useRef(false);

  const selected = parseDate(value);
  const initYear = selected ? selected.getFullYear() : new Date().getFullYear();
  const initMonth = selected ? selected.getMonth() : new Date().getMonth();
  const [viewYear, setViewYear] = useState(initYear);
  const [viewMonth, setViewMonth] = useState(initMonth);

  useEffect(() => {
    setMasked(isoToDisplay(value));
    setInputError(false);
  }, [value]);

  useEffect(() => {
    if (open && selected) {
      setViewYear(selected.getFullYear());
      setViewMonth(selected.getMonth());
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const close = useCallback(() => setOpen(false), []);

  const resolvePendingIso = useCallback(() => {
    const digits = (masked || '').replace(/\D/g, '');
    if (digits.length === 0) return value || '';
    if (digits.length === 8) {
      const iso = displayToISO(masked);
      return iso || value || '';
    }
    const suggestion = smartComplete(digits);
    if (suggestion) {
      const iso = displayToISO(suggestion);
      return iso || value || '';
    }
    return value || '';
  }, [masked, value]);

  useImperativeHandle(ref, () => ({
    /** Commit synchrone de la saisie en cours (ex. avant submit sans blur). */
    flushPending: () => {
      const resolved = resolvePendingIso();
      if (resolved !== (value || '')) {
        onChange(resolved);
      }
      return resolved;
    },
  }), [resolvePendingIso, value, onChange]);

  const handleInputChange = (e) => {
    const raw = e.target.value;
    const deleting = isDeleting.current;
    isDeleting.current = false;

    const digits = raw.replace(/\D/g, '').slice(0, 8);
    const newMasked = applyDateMask(digits);
    setInputError(false);
    pendingSel.current = null;

    if (digits.length === 8) {
      const iso = displayToISO(newMasked);
      if (iso) { onChange(iso); setInputError(false); }
      else setInputError(true);
      setMasked(newMasked);
      return;
    }

    if (!deleting && digits.length >= 1 && digits.length <= 7) {
      const suggestion = smartComplete(digits);
      if (suggestion) {
        setMasked(suggestion);
        pendingSel.current = { start: newMasked.length, end: suggestion.length };
        setTimeout(() => {
          if (inputRef.current && pendingSel.current) {
            inputRef.current.setSelectionRange(pendingSel.current.start, pendingSel.current.end);
          }
        }, 0);
        return;
      }
    }

    setMasked(newMasked);
  };

  const handleInputBlur = () => {
    pendingSel.current = null;
    const digits = (masked || '').replace(/\D/g, '');
    if (digits.length === 0) { onChange(''); setInputError(false); return; }
    if (digits.length === 8) {
      const iso = displayToISO(masked);
      if (iso) { onChange(iso); setInputError(false); }
      else setInputError(true);
      return;
    }
    setInputError(true);
  };

  const handleInputKeyDown = (e) => {
    if (e.key === 'Backspace' || e.key === 'Delete') {
      isDeleting.current = true;
      pendingSel.current = null;
    }
    if (e.key === 'Tab' && pendingSel.current) {
      e.preventDefault();
      const digits = (masked || '').replace(/\D/g, '');
      if (digits.length === 8) {
        const iso = displayToISO(masked);
        if (iso) { onChange(iso); setInputError(false); }
      }
      pendingSel.current = null;
      if (inputRef.current) {
        const len = (masked || '').length;
        inputRef.current.setSelectionRange(len, len);
      }
      return;
    }
    if (e.key === 'Enter') { e.preventDefault(); e.target.blur(); }
  };

  const updatePosition = useCallback(() => {
    if (!wrapperRef.current) return;
    const rect = wrapperRef.current.getBoundingClientRect();
    const popW = 222;
    const margin = 8;
    const gap = 6;
    const el = popoverRef.current;
    const measured = el?.offsetHeight;
    const popH = measured && measured > 48 ? measured : 252;
    const spaceBelow = window.innerHeight - rect.bottom - margin;
    const spaceAbove = rect.top - margin;
    let top;
    if (spaceBelow >= popH + gap || spaceBelow >= spaceAbove) {
      top = rect.bottom + gap;
    } else {
      top = rect.top - popH - gap;
    }
    let left = Math.min(rect.left, window.innerWidth - popW - margin);
    left = Math.max(margin, Math.min(left, window.innerWidth - popW - margin));
    top = Math.max(margin, Math.min(top, window.innerHeight - popH - margin));
    setPos({ top, left });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    updatePosition();
    const id = window.requestAnimationFrame(() => updatePosition());
    window.addEventListener('scroll', updatePosition, true);
    window.addEventListener('resize', updatePosition);
    return () => {
      window.cancelAnimationFrame(id);
      window.removeEventListener('scroll', updatePosition, true);
      window.removeEventListener('resize', updatePosition);
    };
  }, [open, viewMonth, viewYear, updatePosition]);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (wrapperRef.current?.contains(e.target) || popoverRef.current?.contains(e.target)) return;
      close();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, close]);

  const prevMonth = () => { if (viewMonth === 0) { setViewMonth(11); setViewYear((y) => y - 1); } else setViewMonth((m) => m - 1); };
  const nextMonth = () => { if (viewMonth === 11) { setViewMonth(0); setViewYear((y) => y + 1); } else setViewMonth((m) => m + 1); };

  const selectDay = (day) => {
    const d = new Date(viewYear, viewMonth, day);
    onChange(formatISO(d));
    close();
  };

  const clearDate = (e) => { e.stopPropagation(); onChange(''); close(); };

  const totalDays = daysInMonth(viewYear, viewMonth);
  const offset = startDayOfWeek(viewYear, viewMonth);
  const today = new Date();
  const todayStr = formatISO(today);
  const cells = [];
  for (let i = 0; i < offset; i++) cells.push(null);
  for (let d = 1; d <= totalDays; d++) cells.push(d);

  return (
    <>
      <div ref={wrapperRef} className={`${dp.field} ${className}`.trim()}>
        <input
          ref={inputRef}
          id={inputId}
          type="text"
          inputMode="numeric"
          className={`form-input ${dp.input} ${inputClassName} ${inputError ? 'error' : ''}`.trim()}
          value={masked || ''}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyDown={handleInputKeyDown}
          placeholder="__.__.____"
          maxLength={10}
          aria-invalid={invalid || inputError}
          aria-label={ariaLabel}
          title={title}
        />
        <button
          type="button"
          className={dp.iconBtn}
          onClick={() => setOpen(!open)}
          tabIndex={-1}
          aria-expanded={open}
          aria-haspopup="dialog"
          aria-label={open ? 'Fermer le sélecteur de date' : 'Ouvrir le sélecteur de date'}
        >
          <FiCalendar size={14} />
        </button>
      </div>

      {open && createPortal(
        <div
          ref={popoverRef}
          className={dp.popover}
          role="dialog"
          aria-modal="true"
          aria-label="Choisir une date"
          style={{ top: pos.top, left: pos.left }}
        >
          <div className={dp.header}>
            <button type="button" className={dp.navBtn} onClick={prevMonth} aria-label="Mois précédent">
              <FiChevronLeft size={14} />
            </button>
            <span className={dp.headerTitle}>{MONTHS[viewMonth]} {viewYear}</span>
            <button type="button" className={dp.navBtn} onClick={nextMonth} aria-label="Mois suivant">
              <FiChevronRight size={14} />
            </button>
          </div>
          <div className={dp.weekRow}>
            {DAYS.map((d) => <span key={d} className={dp.weekDay}>{d}</span>)}
          </div>
          <div className={dp.grid}>
            {cells.map((day, i) => {
              if (day === null) return <span key={`e-${i}`} className={dp.emptyCell} />;
              const iso = formatISO(new Date(viewYear, viewMonth, day));
              const isSelected = value === iso;
              const isToday = iso === todayStr;
              return (
                <button
                  key={day}
                  type="button"
                  className={`${dp.dayCell} ${isSelected ? dp.dayCellSelected : ''} ${isToday && !isSelected ? dp.dayCellToday : ''}`}
                  onClick={() => selectDay(day)}
                >
                  {day}
                </button>
              );
            })}
          </div>
          <div className={dp.footer}>
            <button
              type="button"
              className={dp.footerBtn}
              onClick={() => {
                onChange(formatISO(today));
                close();
              }}
            >
              Aujourd&apos;hui
            </button>
            {value && <button type="button" className={dp.footerBtnClear} onClick={clearDate}>Effacer</button>}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
});

export default InlineDatePicker;
