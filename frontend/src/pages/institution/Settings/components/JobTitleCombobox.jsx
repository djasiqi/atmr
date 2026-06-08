import React, { useEffect, useMemo, useRef, useState } from 'react';
import { JOB_TITLE_OPTIONS, JOB_TITLE_MAX_LENGTH } from '../../../../constants/jobTitles';
import styles from './JobTitleCombobox.module.css';

/**
 * Champ "Fonction / Métier" : saisie libre + menu de suggestions en chips.
 *
 * - Le menu s'ouvre sous l'input et épouse sa largeur (width 100%).
 * - Les suggestions sont filtrées par le texte saisi.
 * - Saisie libre toujours possible (donnée descriptive, sans permission).
 * - `value`/`onChange` : mode contrôlé (formulaire de création).
 * - `onCommit` : appelé au blur / Entrée / sélection (édition inline).
 */
const JobTitleCombobox = ({
  id,
  name = 'job_title',
  ariaLabel = 'Fonction / métier',
  value = '',
  onChange,
  onCommit,
  placeholder = 'Ex. Infirmier(ère)…',
  inputStyle,
  disabled = false,
}) => {
  const [text, setText] = useState(value || '');
  const [open, setOpen] = useState(false);
  const blurTimer = useRef(null);
  const lastCommitted = useRef(value || '');

  useEffect(() => {
    setText(value || '');
    lastCommitted.current = value || '';
  }, [value]);

  useEffect(() => () => clearTimeout(blurTimer.current), []);

  const filtered = useMemo(() => {
    const q = text.trim().toLowerCase();
    if (!q) return JOB_TITLE_OPTIONS;
    return JOB_TITLE_OPTIONS.filter((opt) => opt.toLowerCase().includes(q));
  }, [text]);

  const commit = (val) => {
    const next = val ?? text;
    if (onCommit && next !== lastCommitted.current) {
      lastCommitted.current = next;
      onCommit(next);
    }
  };

  const handleChange = (e) => {
    const v = e.target.value;
    setText(v);
    onChange?.(v);
  };

  const selectOption = (opt) => {
    setText(opt);
    onChange?.(opt);
    setOpen(false);
    commit(opt);
  };

  return (
    <div className={styles.wrapper}>
      <input
        type="text"
        id={id}
        name={name}
        aria-label={ariaLabel}
        autoComplete="off"
        value={text}
        onChange={handleChange}
        onFocus={() => {
          clearTimeout(blurTimer.current);
          setOpen(true);
        }}
        onBlur={() => {
          blurTimer.current = setTimeout(() => {
            setOpen(false);
            commit();
          }, 140);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            setOpen(false);
            commit();
            e.currentTarget.blur();
          } else if (e.key === 'Escape') {
            setOpen(false);
          }
        }}
        placeholder={placeholder}
        maxLength={JOB_TITLE_MAX_LENGTH}
        disabled={disabled}
        title="Fonction / métier (sans impact sur les permissions)"
        style={inputStyle}
      />
      {open && filtered.length > 0 && (
        <div className={styles.chipMenu} role="listbox">
          {filtered.map((opt) => {
            const active = opt.toLowerCase() === text.trim().toLowerCase();
            return (
              <button
                key={opt}
                type="button"
                role="option"
                aria-selected={active}
                className={`${styles.chipOption} ${active ? styles.chipOptionActive : ''}`}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => selectOption(opt)}
              >
                {opt}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default JobTitleCombobox;
