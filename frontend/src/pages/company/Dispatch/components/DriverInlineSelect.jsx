import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FiChevronDown, FiLoader } from 'react-icons/fi';
import styles from './DriverInlineSelect.module.css';

/**
 * Dropdown inline pour assigner un chauffeur directement dans la cellule du tableau.
 * V3: Assignation inline (pas de modale)
 * V4: Charge chauffeur visible
 * V9: Anti double assignation
 * V10: position:fixed pour eviter les coupures
 * V14: Accessibilite clavier
 */
const DriverInlineSelect = ({
  drivers = [],
  reservationId,
  onAssign,
  currentDriverName = null,
  autoOpen = false,
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [assigning, setAssigning] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const triggerRef = useRef(null);
  const dropdownRef = useRef(null);
  const listRef = useRef(null);
  const [dropdownPos, setDropdownPos] = useState({ top: 0, left: 0, width: 0, openAbove: false });

  // V14: auto-open pour mode assignation rapide
  useEffect(() => {
    if (autoOpen && !disabled && !assigning) {
      setIsOpen(true);
    }
  }, [autoOpen, disabled, assigning]);

  // V10: Calculer position fixed via getBoundingClientRect
  // V15: Flip vers le haut si pas assez d'espace en bas
  const updatePosition = useCallback(() => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    const dropdownMaxHeight = 240;
    const margin = 8;
    const spaceBelow = window.innerHeight - rect.bottom - margin;
    const spaceAbove = rect.top - margin;
    const openAbove = spaceBelow < dropdownMaxHeight && spaceAbove > spaceBelow;

    setDropdownPos({
      top: openAbove ? rect.top - margin : rect.bottom + 4,
      left: rect.left,
      width: Math.max(rect.width, 220),
      openAbove,
    });
  }, []);

  useEffect(() => {
    if (isOpen) {
      updatePosition();
      setActiveIndex(-1);
    }
  }, [isOpen, updatePosition]);

  // Recalculer la position au scroll/resize pour garder le dropdown aligne
  useEffect(() => {
    if (!isOpen) return;
    window.addEventListener('scroll', updatePosition, true);
    window.addEventListener('resize', updatePosition);
    return () => {
      window.removeEventListener('scroll', updatePosition, true);
      window.removeEventListener('resize', updatePosition);
    };
  }, [isOpen, updatePosition]);

  // Click outside pour fermer
  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e) => {
      if (
        triggerRef.current && !triggerRef.current.contains(e.target) &&
        dropdownRef.current && !dropdownRef.current.contains(e.target)
      ) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // V9: Anti double assignation + optimistic UI
  const handleSelect = useCallback(async (driver) => {
    if (assigning || disabled) return;
    setAssigning(true);
    setIsOpen(false);

    try {
      await onAssign(reservationId, driver.id);
    } finally {
      setAssigning(false);
    }
  }, [assigning, disabled, onAssign, reservationId]);

  // V14: Navigation clavier
  const handleKeyDown = useCallback((e) => {
    if (!isOpen) {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
        e.preventDefault();
        setIsOpen(true);
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setActiveIndex((prev) => Math.min(prev + 1, drivers.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setActiveIndex((prev) => Math.max(prev - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        if (activeIndex >= 0 && activeIndex < drivers.length) {
          handleSelect(drivers[activeIndex]);
        }
        break;
      case 'Escape':
        e.preventDefault();
        setIsOpen(false);
        triggerRef.current?.focus();
        break;
      default:
        break;
    }
  }, [isOpen, activeIndex, drivers, handleSelect]);

  // Scroll vers l'element actif
  useEffect(() => {
    if (activeIndex >= 0 && listRef.current) {
      const activeEl = listRef.current.children[activeIndex];
      if (activeEl) {
        activeEl.scrollIntoView({ block: 'nearest' });
      }
    }
  }, [activeIndex]);

  const toggleOpen = (e) => {
    e.stopPropagation();
    if (disabled || assigning) return;
    setIsOpen((prev) => !prev);
  };

  // Mode assigne : nom cliquable pour reassigner
  if (currentDriverName) {
    return (
      <div className={styles.assignedWrapper}>
        <span
          ref={triggerRef}
          className={`${styles.assignedNameClickable} ${assigning ? styles.assignedNameAssigning : ''}`}
          onClick={toggleOpen}
          onKeyDown={handleKeyDown}
          role="button"
          tabIndex={0}
          aria-expanded={isOpen}
          aria-haspopup="listbox"
          title="Cliquer pour reassigner"
        >
          {assigning ? (
            <><FiLoader size={11} className={styles.spinIcon} /> Assignation...</>
          ) : (
            currentDriverName
          )}
        </span>
        {isOpen && (
          <div
            ref={dropdownRef}
            className={`${styles.dropdown} ${dropdownPos.openAbove ? styles.dropdownAbove : ''}`}
            style={{
              ...(dropdownPos.openAbove
                ? { bottom: window.innerHeight - dropdownPos.top, top: 'auto' }
                : { top: dropdownPos.top }),
              left: dropdownPos.left,
              minWidth: dropdownPos.width,
            }}
            role="listbox"
            aria-label="Choisir un chauffeur"
          >
            <div ref={listRef}>
              {drivers.map((driver, idx) => (
                <div
                  key={driver.id}
                  role="option"
                  aria-selected={idx === activeIndex}
                  className={`${styles.driverOption} ${idx === activeIndex ? styles.driverOptionActive : ''}`}
                  onClick={() => handleSelect(driver)}
                  onMouseEnter={() => setActiveIndex(idx)}
                >
                  <span className={styles.driverName}>
                    {driver.full_name || driver.name || driver.username}
                  </span>
                  <span className={`${styles.driverLoad} ${(driver.courseCount || 0) > 3 ? styles.driverLoadHigh : ''}`}>
                    {driver.courseCount || 0} course{(driver.courseCount || 0) !== 1 ? 's' : ''}
                  </span>
                </div>
              ))}
              {drivers.length === 0 && (
                <div className={styles.noDrivers}>Aucun chauffeur disponible</div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Mode non assigne : badge trigger
  return (
    <div className={styles.wrapper}>
      <button
        ref={triggerRef}
        className={`${styles.trigger} ${assigning ? styles.triggerAssigning : ''}`}
        onClick={toggleOpen}
        onKeyDown={handleKeyDown}
        disabled={disabled || assigning}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        tabIndex={0}
      >
        {assigning ? (
          <>
            <FiLoader size={12} className={styles.spinIcon} />
            Assignation...
          </>
        ) : (
          <>
            Non assigne
            <FiChevronDown size={12} />
          </>
        )}
      </button>
      {isOpen && (
        <div
          ref={dropdownRef}
          className={`${styles.dropdown} ${dropdownPos.openAbove ? styles.dropdownAbove : ''}`}
          style={{
            ...(dropdownPos.openAbove
              ? { bottom: window.innerHeight - dropdownPos.top, top: 'auto' }
              : { top: dropdownPos.top }),
            left: dropdownPos.left,
            minWidth: dropdownPos.width,
          }}
          role="listbox"
          aria-label="Choisir un chauffeur"
        >
          <div ref={listRef}>
            {drivers.map((driver, idx) => (
              <div
                key={driver.id}
                role="option"
                aria-selected={idx === activeIndex}
                className={`${styles.driverOption} ${idx === activeIndex ? styles.driverOptionActive : ''}`}
                onClick={() => handleSelect(driver)}
                onMouseEnter={() => setActiveIndex(idx)}
              >
                <span className={styles.driverName}>
                  {driver.full_name || driver.name || driver.username}
                </span>
                <span className={`${styles.driverLoad} ${(driver.courseCount || 0) > 3 ? styles.driverLoadHigh : ''}`}>
                  {driver.courseCount || 0} course{(driver.courseCount || 0) !== 1 ? 's' : ''}
                </span>
              </div>
            ))}
            {drivers.length === 0 && (
              <div className={styles.noDrivers}>Aucun chauffeur disponible</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default React.memo(DriverInlineSelect);
