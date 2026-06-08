import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FiChevronDown } from 'react-icons/fi';
import styles from './ChipSelect.module.css';

/**
 * Sélecteur générique sous forme de chip menu.
 * Reprend le format du ChipDropdown de ReservationFilterBar (côté transport).
 *
 * - `options` : [{ value, label, desc }]
 * - `block` : déclencheur pleine largeur (formulaire) vs chip compact (tableau/filtre).
 * - `placeholder` : libellé affiché quand aucune valeur ne correspond.
 */
const ChipSelect = ({
  id,
  name,
  ariaLabel = 'Sélection',
  value,
  options = [],
  onChange,
  disabled = false,
  block = false,
  placeholder = 'Sélectionner…',
}) => {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return undefined;
    const onClickOutside = (e) => {
      if (ref.current && !ref.current.contains(e.target)) close();
    };
    const onEsc = (e) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('mousedown', onClickOutside);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onClickOutside);
      document.removeEventListener('keydown', onEsc);
    };
  }, [open, close]);

  const selected = options.find((o) => o.value === value);

  return (
    <div className={`${styles.chipDrop} ${block ? styles.chipDropBlock : ''}`} ref={ref}>
      <button
        type="button"
        id={id}
        name={name}
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
        title={selected?.desc || selected?.label || ''}
        className={`${styles.chipBtn} ${block ? styles.chipBtnBlock : ''}`}
        onClick={() => setOpen((p) => !p)}
      >
        <span className={styles.chipText}>{selected?.label || placeholder}</span>
        <FiChevronDown
          size={12}
          className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`}
        />
      </button>
      {open && (
        <div className={styles.chipMenu} role="listbox">
          {options.map((o) => (
            <button
              key={o.value}
              type="button"
              role="option"
              aria-selected={o.value === value}
              title={o.desc || ''}
              className={`${styles.chipOption} ${o.value === value ? styles.chipOptionActive : ''}`}
              onClick={() => {
                onChange(o.value);
                close();
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default ChipSelect;
