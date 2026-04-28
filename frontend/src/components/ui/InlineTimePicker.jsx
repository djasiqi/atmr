import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { FiClock } from 'react-icons/fi';
import tp from './InlineTimePicker.module.css';

const pad = (n) => String(n).padStart(2, '0');

const PRESETS = ['06:00', '07:00', '08:00', '09:00', '10:00', '12:00', '14:00', '16:00', '18:00'];

function parseTime(val) {
  if (!val) return { h: '', m: '' };
  const [h, m] = val.split(':');
  return { h: h || '', m: m || '' };
}

export default function InlineTimePicker({ value, onChange, placeholder: _placeholder, className, inputId, onSelectNow }) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);
  const popoverRef = useRef(null);
  const inputRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const [masked, setMasked] = useState('');
  const [showUndefinedLabel, setShowUndefinedLabel] = useState(false);

  const { h: initH, m: initM } = parseTime(value);
  const [hours, setHours] = useState(initH || '08');
  const [minutes, setMinutes] = useState(initM || '00');

  useEffect(() => {
    if (value) {
      const { h, m } = parseTime(value);
      setMasked(`${h}:${m}`);
      setHours(h);
      setMinutes(m);
      setShowUndefinedLabel(false);
    } else {
      if (!showUndefinedLabel) {
        setMasked('');
      }
    }
  }, [value, showUndefinedLabel]);

  const close = useCallback(() => setOpen(false), []);

  const handleInputChange = (e) => {
    if (showUndefinedLabel) {
      setShowUndefinedLabel(false);
      setMasked('');
    }
    const raw = e.target.value;
    const digits = raw.replace(/\D/g, '').slice(0, 4);
    let display = '';
    for (let i = 0; i < digits.length; i++) {
      if (i === 2) display += ':';
      let d = parseInt(digits[i], 10);
      if (i === 0 && d > 2) d = 2;
      if (i === 1 && parseInt(digits[0], 10) === 2 && d > 3) d = 3;
      if (i === 2 && d > 5) d = 5;
      display += d;
    }
    setMasked(display);
    if (digits.length === 4) {
      const hh = digits.slice(0, 2);
      const mm = digits.slice(2, 4);
      if (parseInt(hh, 10) <= 23 && parseInt(mm, 10) <= 59) {
        setHours(hh); setMinutes(mm); onChange(`${hh}:${mm}`);
      }
    }
  };

  const handleInputFocus = () => {
    if (showUndefinedLabel) {
      setShowUndefinedLabel(false);
      setMasked('');
    }
  };

  const handleInputBlur = () => {
    if (showUndefinedLabel) return;
    const digits = masked.replace(/\D/g, '');
    if (digits.length === 0) { onChange(''); return; }
    if (digits.length < 4) setMasked(value ? `${parseTime(value).h}:${parseTime(value).m}` : '');
  };

  const handleInputKeyDown = (e) => {
    if (e.key === 'Enter') { e.preventDefault(); e.target.blur(); }
  };

  const selectPreset = (preset) => {
    const { h, m } = parseTime(preset);
    setHours(h); setMinutes(m); setMasked(`${h}:${m}`);
    setShowUndefinedLabel(false);
    onChange(`${h}:${m}`);
    close();
  };

  const onHoursChange = (e) => { const h = e.target.value; setHours(h); setMasked(`${h}:${minutes}`); setShowUndefinedLabel(false); onChange(`${h}:${minutes}`); };
  const onMinutesChange = (e) => { const m = e.target.value; setMinutes(m); setMasked(`${hours}:${m}`); setShowUndefinedLabel(false); onChange(`${hours}:${m}`); };

  const updatePosition = useCallback(() => {
    if (!wrapperRef.current) return;
    const rect = wrapperRef.current.getBoundingClientRect();
    const popH = 220, popW = 180;
    let top = rect.bottom + window.scrollY + 4;
    let left = rect.left + window.scrollX;
    if (rect.bottom + popH > window.innerHeight) top = rect.top + window.scrollY - popH - 4;
    if (left + popW > window.innerWidth) left = window.innerWidth - popW - 8;
    setPos({ top, left });
  }, []);

  useEffect(() => {
    if (!open) return;
    updatePosition();
    window.addEventListener('scroll', updatePosition, true);
    window.addEventListener('resize', updatePosition);
    return () => { window.removeEventListener('scroll', updatePosition, true); window.removeEventListener('resize', updatePosition); };
  }, [open, updatePosition]);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (popoverRef.current?.contains(e.target) || wrapperRef.current?.contains(e.target)) return;
      close();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, close]);

  const currentVal = value ? `${parseTime(value).h}:${parseTime(value).m}` : null;

  return (
    <>
      <div ref={wrapperRef} className={tp.field}>
        <input
          ref={inputRef}
          id={inputId}
          type="text"
          inputMode="numeric"
          className={`form-input ${tp.input} ${showUndefinedLabel ? tp.inputUndefined : ''} ${className || ''}`}
          value={masked}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onBlur={handleInputBlur}
          onKeyDown={handleInputKeyDown}
          placeholder="__:__"
          maxLength={5}
        />
        <button
          type="button"
          className={tp.iconBtn}
          onClick={() => setOpen(!open)}
          tabIndex={-1}
          aria-label={open ? 'Fermer le sélecteur d’heure' : 'Ouvrir le sélecteur d’heure'}
        >
          <FiClock size={14} />
        </button>
      </div>

      {open && createPortal(
        <div ref={popoverRef} className={tp.popover} style={{ top: pos.top, left: pos.left }}>
          <div className={tp.popoverTitle}>
            <FiClock size={11} className={tp.popoverTitleIcon} />
            Heure
          </div>
          <div className={tp.selectRow}>
            <select className={tp.select} value={hours} onChange={onHoursChange} aria-label="Heures">
              {Array.from({ length: 24 }, (_, i) => <option key={i} value={pad(i)}>{pad(i)}</option>)}
            </select>
            <span className={tp.sep}>:</span>
            <select className={tp.select} value={minutes} onChange={onMinutesChange} aria-label="Minutes">
              {Array.from({ length: 12 }, (_, i) => i * 5).map((m) => <option key={m} value={pad(m)}>{pad(m)}</option>)}
            </select>
          </div>
          <div className={tp.presets}>
            {PRESETS.map((p) => (
              <button key={p} type="button" className={`${tp.preset} ${currentVal === p ? tp.presetActive : ''}`} onClick={() => selectPreset(p)}>{p}</button>
            ))}
          </div>
          <div className={tp.footer}>
            <button
              type="button"
              className={tp.footerBtn}
              onClick={() => {
                const now = new Date();
                if (typeof onSelectNow === 'function') {
                  onSelectNow(now);
                }
                selectPreset(`${pad(now.getHours())}:${pad(Math.ceil(now.getMinutes() / 5) * 5 % 60)}`);
              }}
            >
              Maintenant
            </button>
            <button type="button" className={tp.footerBtn} onClick={() => { onChange(''); setMasked('A definir'); setShowUndefinedLabel(true); close(); }}>À définir</button>
            {(value || showUndefinedLabel) && <button type="button" className={tp.footerBtnClear} onClick={() => { onChange(''); setMasked(''); setShowUndefinedLabel(false); close(); }}>Effacer</button>}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}
