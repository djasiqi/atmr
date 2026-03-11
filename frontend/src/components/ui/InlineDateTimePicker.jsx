import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { FiChevronLeft, FiChevronRight, FiCalendar, FiClock } from 'react-icons/fi';
import dp from './InlineDatePicker.module.css';
import dtp from './InlineDateTimePicker.module.css';

const DAYS = ['Lu', 'Ma', 'Me', 'Je', 'Ve', 'Sa', 'Di'];
const MONTHS = [
  'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
  'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre',
];

const pad = (n) => String(n).padStart(2, '0');

function parseDT(str) {
  if (!str) return { date: null, hours: '08', minutes: '00' };
  const [datePart, timePart] = str.split('T');
  const [h, m] = (timePart || '08:00').split(':');
  return { date: datePart || null, hours: h || '08', minutes: m || '00' };
}

function formatISO(y, m, d) {
  return `${y}-${pad(m + 1)}-${pad(d)}`;
}

function daysInMonth(year, month) {
  return new Date(year, month + 1, 0).getDate();
}

function startDayOfWeek(year, month) {
  const day = new Date(year, month, 1).getDay();
  return day === 0 ? 6 : day - 1;
}

const MASK = '__.__.____ __:__';

function digitsFromDisplay(str) {
  return (str || '').replace(/\D/g, '');
}

function clampDigit(digits, pos, digit) {
  const d = parseInt(digit, 10);
  switch (pos) {
    case 0: return d > 3 ? '3' : digit;
    case 1: {
      const d0 = parseInt(digits[0], 10);
      if (d0 === 3) return d > 1 ? '1' : digit;
      return digit;
    }
    case 2: return d > 1 ? '1' : digit;
    case 3: {
      const m0 = parseInt(digits[2], 10);
      if (m0 === 1) return d > 2 ? '2' : digit;
      if (m0 === 0) return d === 0 ? '1' : digit;
      return digit;
    }
    case 4: return d !== 2 ? '2' : digit;
    case 5: return d !== 0 ? '0' : digit;
    case 6: {
      const y01 = digits.slice(4, 6);
      if (y01 === '20') return d > 3 ? '3' : digit;
      return digit;
    }
    case 7: return digit;
    case 8: return d > 2 ? '2' : digit;
    case 9: {
      const h0 = parseInt(digits[8], 10);
      if (h0 === 2) return d > 3 ? '3' : digit;
      return digit;
    }
    case 10: return d > 5 ? '5' : digit;
    case 11: return digit;
    default: return digit;
  }
}

function applyMask(rawDigits) {
  const clamped = [];
  for (let i = 0; i < rawDigits.length && i < 12; i++) {
    clamped.push(clampDigit(clamped, i, rawDigits[i]));
  }
  let out = '';
  for (let i = 0; i < clamped.length; i++) {
    if (i === 2 || i === 4) out += '.';
    else if (i === 8) out += ' ';
    else if (i === 10) out += ':';
    out += clamped[i];
  }
  return out;
}

function maskedToISO(masked) {
  const d = digitsFromDisplay(masked);
  if (d.length !== 12) return null;
  const dd = parseInt(d.slice(0, 2), 10);
  const mm = parseInt(d.slice(2, 4), 10);
  const yyyy = parseInt(d.slice(4, 8), 10);
  const hh = parseInt(d.slice(8, 10), 10);
  const mi = parseInt(d.slice(10, 12), 10);
  if (mm < 1 || mm > 12 || dd < 1 || dd > 31 || hh > 23 || mi > 59 || yyyy < 2025) return null;
  const maxDay = new Date(yyyy, mm, 0).getDate();
  if (dd > maxDay) return null;
  const dt = new Date(yyyy, mm - 1, dd, hh, mi);
  if (dt < new Date()) return null;
  return `${yyyy}-${pad(mm)}-${pad(dd)}T${pad(hh)}:${pad(mi)}`;
}

function smartComplete(digits) {
  const now = new Date();
  const todayMonth = now.getMonth();
  const todayYear = now.getFullYear();

  if (digits.length >= 1 && digits.length <= 2) {
    const targetDay = parseInt(digits.length === 1 ? digits + '0' : digits, 10);
    if (digits.length === 1 && targetDay === 0) return null;
    const day = digits.length === 1 ? parseInt(digits, 10) * 10 : targetDay;
    if (day < 1 || day > 31) return null;

    let m = todayMonth;
    let y = todayYear;
    for (let tries = 0; tries < 13; tries++) {
      const maxD = new Date(y, m + 1, 0).getDate();
      if (day <= maxD) {
        const endOfDay = new Date(y, m, day, 23, 59);
        if (endOfDay > now) {
          return { dd: pad(day), mm: pad(m + 1), yyyy: String(y) };
        }
      }
      m++;
      if (m > 11) { m = 0; y++; }
    }
    return null;
  }

  if (digits.length >= 3 && digits.length <= 4) {
    const dd = parseInt(digits.slice(0, 2), 10);
    const mmPart = digits.slice(2);
    const mm = mmPart.length === 1 ? parseInt(mmPart + '0', 10) : parseInt(mmPart, 10);
    const month = mmPart.length === 1 ? parseInt(mmPart, 10) * 10 : mm;
    if (dd < 1 || dd > 31 || month < 1 || month > 12) return null;
    let y = todayYear;
    const maxD = new Date(y, month, 0).getDate();
    if (dd > maxD) return null;
    const endOfDay = new Date(y, month - 1, dd, 23, 59);
    if (endOfDay <= now) y++;
    return { dd: pad(dd), mm: pad(month), yyyy: String(y) };
  }

  if (digits.length >= 5 && digits.length <= 8) {
    const dd = parseInt(digits.slice(0, 2), 10);
    const mm = parseInt(digits.slice(2, 4), 10);
    const yyyyStr = digits.slice(4, 8).padEnd(4, '0');
    const yyyy = parseInt(yyyyStr, 10);
    if (dd < 1 || dd > 31 || mm < 1 || mm > 12 || yyyy < 2025) return null;
    const maxD = new Date(yyyy, mm, 0).getDate();
    if (dd > maxD) return null;
    return { dd: pad(dd), mm: pad(mm), yyyy: String(yyyy) };
  }

  return null;
}

function valueToMasked(val) {
  if (!val) return '';
  const { date, hours, minutes } = parseDT(val);
  if (!date) return '';
  const [y, m, d] = date.split('-');
  return `${d}.${m}.${y} ${hours}:${minutes}`;
}

export default function InlineDateTimePicker({ value, onChange, placeholder: _placeholder }) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);
  const popoverRef = useRef(null);
  const inputRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const [masked, setMasked] = useState(() => valueToMasked(value));
  const [inputError, setInputError] = useState(false);
  const pendingSuggestion = useRef(null);

  const { date: selectedDate, hours: initH, minutes: initM } = parseDT(value);
  const [hours, setHours] = useState(initH);
  const [minutes, setMinutes] = useState(initM);

  const now = new Date();
  const initYear = selectedDate ? parseInt(selectedDate.split('-')[0], 10) : now.getFullYear();
  const initMonth = selectedDate ? parseInt(selectedDate.split('-')[1], 10) - 1 : now.getMonth();
  const [viewYear, setViewYear] = useState(initYear);
  const [viewMonth, setViewMonth] = useState(initMonth);

  useEffect(() => {
    setMasked(valueToMasked(value));
    setInputError(false);
    pendingSuggestion.current = null;
  }, [value]);

  useEffect(() => {
    if (open) {
      const { date, hours: h, minutes: m } = parseDT(value);
      setHours(h);
      setMinutes(m);
      if (date) {
        setViewYear(parseInt(date.split('-')[0], 10));
        setViewMonth(parseInt(date.split('-')[1], 10) - 1);
      }
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const close = useCallback(() => setOpen(false), []);

  const handleInputChange = (e) => {
    const raw = e.target.value;
    const cursorPos = e.target.selectionStart;
    const prevLen = (masked || '').length;
    const isTypingForward = cursorPos >= prevLen;

    const digits = digitsFromDisplay(raw).slice(0, 12);
    const newMasked = applyMask(digits);
    setInputError(false);
    pendingSuggestion.current = null;

    if (digits.length === 12) {
      const iso = maskedToISO(newMasked);
      if (iso) { onChange(iso); setInputError(false); }
      else setInputError(true);
      setMasked(newMasked);
      return;
    }

    if (isTypingForward && digits.length >= 2 && digits.length <= 7) {
      const result = smartComplete(digits);
      if (result) {
        const full = `${result.dd}.${result.mm}.${result.yyyy}`;
        setMasked(full);
        pendingSuggestion.current = { selStart: newMasked.length, selEnd: full.length };
        setTimeout(() => {
          if (inputRef.current && pendingSuggestion.current) {
            inputRef.current.setSelectionRange(
              pendingSuggestion.current.selStart,
              pendingSuggestion.current.selEnd,
            );
          }
        }, 0);
        return;
      }
    }

    setMasked(newMasked);
  };

  const handleInputBlur = () => {
    pendingSuggestion.current = null;
    const digits = digitsFromDisplay(masked);
    if (digits.length === 0) {
      onChange('');
      setInputError(false);
      return;
    }
    if (digits.length === 12) {
      const iso = maskedToISO(masked);
      if (iso) { onChange(iso); setInputError(false); }
      else setInputError(true);
      return;
    }
    if (digits.length === 8) {
      setMasked(masked + ' ');
      setInputError(false);
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
          const p = masked.length + 1;
          inputRef.current.setSelectionRange(p, p);
        }
      }, 0);
      return;
    }
    setInputError(true);
  };

  const handleInputKeyDown = (e) => {
    if (e.key === 'Tab' && pendingSuggestion.current) {
      e.preventDefault();
      const full = masked;
      setMasked(full + ' ');
      pendingSuggestion.current = null;
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
          const p = full.length + 1;
          inputRef.current.setSelectionRange(p, p);
        }
      }, 0);
      return;
    }
    if (e.key === 'Enter') { e.preventDefault(); e.target.blur(); }
  };

  const updatePosition = useCallback(() => {
    if (!wrapperRef.current) return;
    const rect = wrapperRef.current.getBoundingClientRect();
    const popH = 360;
    const popW = 240;
    const spaceBelow = window.innerHeight - rect.bottom;
    const top = spaceBelow >= popH ? rect.bottom + 4 : rect.top - popH - 4;
    const left = Math.min(rect.left, window.innerWidth - popW - 8);
    setPos({ top: top + window.scrollY, left: Math.max(8, left + window.scrollX) });
  }, []);

  useEffect(() => {
    if (!open) return;
    updatePosition();
    const onScroll = () => updatePosition();
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onScroll);
    };
  }, [open, updatePosition]);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (
        wrapperRef.current && !wrapperRef.current.contains(e.target) &&
        popoverRef.current && !popoverRef.current.contains(e.target)
      ) close();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, close]);

  const prevMonth = () => {
    if (viewMonth === 0) { setViewMonth(11); setViewYear((y) => y - 1); }
    else setViewMonth((m) => m - 1);
  };

  const nextMonth = () => {
    if (viewMonth === 11) { setViewMonth(0); setViewYear((y) => y + 1); }
    else setViewMonth((m) => m + 1);
  };

  const emit = (dateStr, h, m) => {
    onChange(`${dateStr}T${pad(parseInt(h, 10))}:${pad(parseInt(m, 10))}`);
  };

  const selectDay = (day) => {
    const dateStr = formatISO(viewYear, viewMonth, day);
    emit(dateStr, hours, minutes);
  };

  const onHoursChange = (e) => {
    const h = e.target.value;
    setHours(h);
    if (selectedDate) emit(selectedDate, h, minutes);
  };

  const onMinutesChange = (e) => {
    const m = e.target.value;
    setMinutes(m);
    if (selectedDate) emit(selectedDate, hours, m);
  };

  const clearAll = (e) => {
    e.stopPropagation();
    onChange('');
    close();
  };

  const totalDays = daysInMonth(viewYear, viewMonth);
  const offset = startDayOfWeek(viewYear, viewMonth);
  const todayStr = formatISO(now.getFullYear(), now.getMonth(), now.getDate());

  const cells = [];
  for (let i = 0; i < offset; i++) cells.push(null);
  for (let d = 1; d <= totalDays; d++) cells.push(d);

  return (
    <>
      <div ref={wrapperRef} className={`${dtp.inputWrapper} ${inputError ? dtp.inputWrapperError : ''}`}>
        <input
          ref={inputRef}
          type="text"
          inputMode="numeric"
          className={dtp.inputField}
          value={masked || ''}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyDown={handleInputKeyDown}
          placeholder={MASK}
          maxLength={16}
        />
        <button
          type="button"
          className={dtp.calBtn}
          onClick={() => setOpen(!open)}
          tabIndex={-1}
          aria-label={open ? 'Fermer le sélecteur de date et heure' : 'Ouvrir le sélecteur de date et heure'}
        >
          <FiCalendar size={13} />
        </button>
      </div>

      {open && createPortal(
        <div
          ref={popoverRef}
          className={dp.popover}
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
              const iso = formatISO(viewYear, viewMonth, day);
              const isSelected = selectedDate === iso;
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

          <div className={dtp.timeRow}>
            <FiClock size={11} className={dtp.timeIcon} />
            <span className={dtp.timeLabel}>Heure</span>
            <select className={dtp.timeSelect} value={hours} onChange={onHoursChange}>
              {Array.from({ length: 24 }, (_, i) => (
                <option key={i} value={pad(i)}>{pad(i)}</option>
              ))}
            </select>
            <span className={dtp.timeSep}>:</span>
            <select className={dtp.timeSelect} value={minutes} onChange={onMinutesChange}>
              {[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map((m) => (
                <option key={m} value={pad(m)}>{pad(m)}</option>
              ))}
            </select>
          </div>

          <div className={dp.footer}>
            <button type="button" className={dp.footerBtn} onClick={() => {
              const dateStr = formatISO(now.getFullYear(), now.getMonth(), now.getDate());
              const h = pad(now.getHours());
              const m = pad(Math.ceil(now.getMinutes() / 5) * 5);
              setHours(h);
              setMinutes(m);
              setViewYear(now.getFullYear());
              setViewMonth(now.getMonth());
              emit(dateStr, h, m);
            }}>
              Maintenant
            </button>
            {value && (
              <button type="button" className={dp.footerBtnClear} onClick={clearAll}>
                Effacer
              </button>
            )}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}
