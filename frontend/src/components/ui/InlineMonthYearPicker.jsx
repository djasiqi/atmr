import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { FiChevronLeft, FiChevronRight, FiCalendar } from 'react-icons/fi';
import my from './InlineMonthYearPicker.module.css';

const MONTHS = [
  'Janvier',
  'Février',
  'Mars',
  'Avril',
  'Mai',
  'Juin',
  'Juillet',
  'Août',
  'Septembre',
  'Octobre',
  'Novembre',
  'Décembre',
];

const MIN_YEAR = 2000;
const MAX_YEAR = 2100;

function parseYm(str) {
  if (!str || typeof str !== 'string') return null;
  const [ys, ms] = str.split('-');
  const y = parseInt(ys, 10);
  const m = parseInt(ms, 10);
  if (!Number.isFinite(y) || !Number.isFinite(m) || m < 1 || m > 12) return null;
  if (y < MIN_YEAR || y > MAX_YEAR) return null;
  return { year: y, month: m };
}

function formatYm(year, month) {
  return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}`;
}

function labelFromYm(str) {
  const p = parseYm(str);
  if (!p) return '';
  return `${MONTHS[p.month - 1]} ${p.year}`;
}

/**
 * Sélecteur mois + année avec boutons Effacer et Ce mois (comportement type Flowbite datepicker-buttons).
 * Valeur : chaîne `YYYY-MM` ou chaîne vide après effacement (géré par le parent).
 */
export default function InlineMonthYearPicker({
  value,
  onChange,
  disabled = false,
  inputId = 'inline-month-year',
  className = '',
  inputClassName = '',
  invalid = false,
  ariaLabel,
  title,
}) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);
  const popoverRef = useRef(null);
  const inputRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  const parsed = useMemo(() => parseYm(value), [value]);
  const now = new Date();
  const currentYear = now.getFullYear();
  const currentMonth = now.getMonth() + 1;

  const initYear = parsed?.year ?? currentYear;
  const [viewYear, setViewYear] = useState(() =>
    Math.min(MAX_YEAR, Math.max(MIN_YEAR, initYear))
  );
  const [yearPanelOpen, setYearPanelOpen] = useState(false);
  const [yearGridDecade, setYearGridDecade] = useState(() =>
    Math.floor(Math.min(MAX_YEAR, Math.max(MIN_YEAR, initYear)) / 10) * 10
  );

  useEffect(() => {
    if (parsed) {
      setViewYear(Math.min(MAX_YEAR, Math.max(MIN_YEAR, parsed.year)));
    }
  }, [parsed?.year, value]);

  const minDecadeStart = Math.floor(MIN_YEAR / 10) * 10;
  const maxDecadeStart = Math.floor(MAX_YEAR / 10) * 10;

  const decadeYearList = useMemo(() => {
    const start = yearGridDecade;
    return Array.from({ length: 10 }, (_, i) => start + i).filter(
      (y) => y >= MIN_YEAR && y <= MAX_YEAR
    );
  }, [yearGridDecade]);

  const close = useCallback(() => {
    setOpen(false);
    setYearPanelOpen(false);
  }, []);

  useEffect(() => {
    if (!open) setYearPanelOpen(false);
  }, [open]);

  const updatePosition = useCallback(() => {
    if (!wrapperRef.current) return;
    const rect = wrapperRef.current.getBoundingClientRect();
    const popH = 320;
    const popW = 268;
    const margin = 8;
    const spaceBelow = window.innerHeight - rect.bottom;
    let top = spaceBelow >= popH ? rect.bottom + 4 : rect.top - popH - 4;
    let left = Math.min(rect.left, window.innerWidth - popW - margin);
    left = Math.max(margin, Math.min(left, window.innerWidth - popW - margin));
    top = Math.max(margin, Math.min(top, window.innerHeight - popH - margin));
    setPos({ top, left });
  }, []);

  useEffect(() => {
    if (!open) return;
    updatePosition();
    window.addEventListener('scroll', updatePosition, true);
    window.addEventListener('resize', updatePosition);
    return () => {
      window.removeEventListener('scroll', updatePosition, true);
      window.removeEventListener('resize', updatePosition);
    };
  }, [open, updatePosition]);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (wrapperRef.current?.contains(e.target) || popoverRef.current?.contains(e.target)) return;
      close();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, close]);

  const selectMonth = (m) => {
    const y = Math.min(MAX_YEAR, Math.max(MIN_YEAR, viewYear));
    onChange(formatYm(y, m));
    close();
  };

  const handleToday = () => {
    const y = currentYear;
    const m = currentMonth;
    setViewYear(Math.min(MAX_YEAR, Math.max(MIN_YEAR, y)));
    onChange(formatYm(y, m));
    close();
  };

  const handleClear = (e) => {
    e.stopPropagation();
    onChange('');
    close();
  };

  const prevYear = () => setViewYear((y) => Math.max(MIN_YEAR, y - 1));
  const nextYear = () => setViewYear((y) => Math.min(MAX_YEAR, y + 1));

  const openYearPanel = () => {
    setYearGridDecade(Math.floor(viewYear / 10) * 10);
    setYearPanelOpen(true);
  };

  const prevDecade = () =>
    setYearGridDecade((d) => Math.max(minDecadeStart, d - 10));
  const nextDecade = () =>
    setYearGridDecade((d) => Math.min(maxDecadeStart, d + 10));

  const selectYearFromGrid = (y) => {
    setViewYear(Math.min(MAX_YEAR, Math.max(MIN_YEAR, y)));
    setYearPanelOpen(false);
  };

  const display = labelFromYm(value);

  return (
    <>
      <div ref={wrapperRef} className={`${my.field} ${className}`.trim()}>
        <span className={my.iconLeft} aria-hidden>
          <FiCalendar size={14} strokeWidth={2} />
        </span>
        <input
          ref={inputRef}
          id={inputId}
          type="text"
          readOnly
          tabIndex={disabled ? -1 : 0}
          className={`${my.input} ${inputClassName} ${invalid ? my.inputInvalid : ''}`.trim()}
          value={display}
          placeholder="Mois et année"
          disabled={disabled}
          aria-expanded={open}
          aria-haspopup="dialog"
          aria-invalid={invalid}
          aria-label={ariaLabel}
          title={title}
          onClick={() => !disabled && setOpen((o) => !o)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              if (!disabled) setOpen((o) => !o);
            }
            if (e.key === 'Escape') close();
          }}
        />
      </div>

      {open &&
        createPortal(
          <div
            ref={popoverRef}
            className={my.popover}
            style={{ top: pos.top, left: pos.left }}
            role="dialog"
            aria-modal="true"
            aria-label="Choisir le mois et l’année"
          >
            <div className={my.header}>
              <button
                type="button"
                className={my.navBtn}
                onClick={yearPanelOpen ? prevDecade : prevYear}
                disabled={
                  yearPanelOpen ? yearGridDecade <= minDecadeStart : viewYear <= MIN_YEAR
                }
                aria-label={yearPanelOpen ? 'Décennie précédente' : 'Année précédente'}
              >
                <FiChevronLeft size={16} />
              </button>
              {yearPanelOpen ? (
                <span className={my.decadeTitle} aria-live="polite">
                  {yearGridDecade} – {yearGridDecade + 9}
                </span>
              ) : (
                <button
                  type="button"
                  className={my.yearTitleBtn}
                  onClick={openYearPanel}
                  aria-expanded={yearPanelOpen}
                  aria-haspopup="grid"
                  aria-label={`Choisir l’année, affichage ${viewYear}`}
                >
                  {viewYear}
                </button>
              )}
              <button
                type="button"
                className={my.navBtn}
                onClick={yearPanelOpen ? nextDecade : nextYear}
                disabled={
                  yearPanelOpen ? yearGridDecade >= maxDecadeStart : viewYear >= MAX_YEAR
                }
                aria-label={yearPanelOpen ? 'Décennie suivante' : 'Année suivante'}
              >
                <FiChevronRight size={16} />
              </button>
            </div>
            {yearPanelOpen && (
              <button
                type="button"
                className={my.backToMonths}
                onClick={() => setYearPanelOpen(false)}
              >
                ← Mois
              </button>
            )}
            {yearPanelOpen ? (
              <div className={my.yearGrid} role="grid" aria-label="Choisir une année">
                {decadeYearList.map((y) => {
                  const selected = parsed?.year === y;
                  const isThisYear = currentYear === y;
                  return (
                    <button
                      key={y}
                      type="button"
                      role="gridcell"
                      className={`${my.yearCell} ${selected ? my.yearCellSelected : ''} ${
                        !selected && isThisYear ? my.yearCellCurrent : ''
                      }`.trim()}
                      onClick={() => selectYearFromGrid(y)}
                    >
                      {y}
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className={my.monthGrid}>
                {MONTHS.map((name, i) => {
                  const m = i + 1;
                  const selected =
                    parsed && parsed.year === viewYear && parsed.month === m;
                  const isThisMonth = currentYear === viewYear && currentMonth === m;
                  return (
                    <button
                      key={name}
                      type="button"
                      className={`${my.monthCell} ${selected ? my.monthCellSelected : ''} ${
                        !selected && isThisMonth ? my.monthCellCurrent : ''
                      }`.trim()}
                      onClick={() => selectMonth(m)}
                    >
                      {name}
                    </button>
                  );
                })}
              </div>
            )}
            <div className={my.footer}>
              <button type="button" className={my.footerBtnClear} onClick={handleClear}>
                Effacer
              </button>
              <button
                type="button"
                className={my.footerBtnToday}
                onClick={handleToday}
                title="Sélectionner le mois civil en cours"
                aria-label="Sélectionner le mois en cours"
              >
                Ce mois
              </button>
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
