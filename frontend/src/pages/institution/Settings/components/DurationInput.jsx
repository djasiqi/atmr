import React, { useEffect, useRef, useState } from 'react';
import {
  DURATION_UNITS,
  displayValueToMinutes,
  formatDurationLabel,
  formatDurationRangeHint,
  getDurationBounds,
  minutesToDisplayValue,
  pickDefaultDurationUnit,
} from '../../../../utils/durationInput';
import styles from '../InstitutionSettings.module.css';

/**
 * Saisie de durée avec bascule minutes / heures.
 * La valeur exposée via onChange reste toujours en minutes (API).
 */
const DurationInput = ({
  id,
  label,
  value,
  onChange,
  minMinutes = 1,
  maxMinutes = 10080,
  recommendedMinutes,
  disabled = false,
  hint,
}) => {
  const [unit, setUnit] = useState(() => pickDefaultDurationUnit(value));
  const [displayValue, setDisplayValue] = useState(() =>
    String(minutesToDisplayValue(value, pickDefaultDurationUnit(value))),
  );

  // Resynchroniser uniquement quand la valeur vient de l'extérieur (chargement serveur, reset).
  const lastEmittedMinutes = useRef(value);
  useEffect(() => {
    if (value === lastEmittedMinutes.current) return;
    lastEmittedMinutes.current = value;
    const nextUnit = pickDefaultDurationUnit(value);
    setUnit(nextUnit);
    setDisplayValue(String(minutesToDisplayValue(value, nextUnit)));
  }, [value]);

  const bounds = getDurationBounds(minMinutes, maxMinutes, unit);
  const minutesEquivalent = displayValueToMinutes(displayValue, unit);
  const isOutOfRange =
    minutesEquivalent == null ||
    minutesEquivalent < minMinutes ||
    minutesEquivalent > maxMinutes;

  const handleValueChange = (raw) => {
    setDisplayValue(raw);
    const minutes = displayValueToMinutes(raw, unit);
    if (minutes != null && minutes >= minMinutes && minutes <= maxMinutes) {
      lastEmittedMinutes.current = minutes;
      onChange(minutes);
    }
  };

  const handleUnitChange = (nextUnit) => {
    const currentMinutes = displayValueToMinutes(displayValue, unit);
    setUnit(nextUnit);
    if (currentMinutes != null) {
      const clamped = Math.min(maxMinutes, Math.max(minMinutes, currentMinutes));
      setDisplayValue(String(minutesToDisplayValue(clamped, nextUnit)));
      lastEmittedMinutes.current = clamped;
      onChange(clamped);
    } else {
      setDisplayValue(String(minutesToDisplayValue(value, nextUnit)));
    }
  };

  const rangeHint = formatDurationRangeHint(minMinutes, maxMinutes, unit);
  const recommendedLabel =
    recommendedMinutes != null ? formatDurationLabel(recommendedMinutes) : null;
  const showHourEquivalent =
    unit === DURATION_UNITS.MINUTES &&
    minutesEquivalent != null &&
    !isOutOfRange &&
    minutesEquivalent >= 60;

  return (
    <div className={styles.field}>
      <label htmlFor={id}>{label}</label>
      <div className={styles.durationInputGroup}>
        <input
          id={id}
          type="number"
          min={bounds.min}
          max={bounds.max}
          step={bounds.step}
          value={displayValue}
          onChange={(e) => handleValueChange(e.target.value)}
          disabled={disabled}
          aria-invalid={isOutOfRange || undefined}
          className={isOutOfRange ? styles.durationInputInvalid : undefined}
        />
        <select
          className={styles.durationUnitSelect}
          value={unit}
          onChange={(e) => handleUnitChange(e.target.value)}
          disabled={disabled}
          aria-label={`Unité pour ${label}`}
        >
          <option value={DURATION_UNITS.MINUTES}>min</option>
          <option value={DURATION_UNITS.HOURS}>h</option>
        </select>
      </div>
      <span className={styles.fieldHint}>
        {hint || `Délai accordé au transporteur (${rangeHint}).`}
        {recommendedLabel && <> Recommandé : {recommendedLabel}.</>}
        {showHourEquivalent && <> Soit {formatDurationLabel(minutesEquivalent)}.</>}
        {isOutOfRange && displayValue !== '' && (
          <> Valeur hors plage ({rangeHint}).</>
        )}
      </span>
    </div>
  );
};

export default DurationInput;
