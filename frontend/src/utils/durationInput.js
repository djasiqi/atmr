/** Unités affichables pour les champs de durée (stockage API toujours en minutes). */
export const DURATION_UNITS = {
  MINUTES: 'minutes',
  HOURS: 'hours',
};

/** Choisit l'unité la plus lisible : minutes jusqu'à 60, heures au-delà. */
export function pickDefaultDurationUnit(minutes) {
  const value = Number(minutes);
  if (!Number.isFinite(value) || value <= 60) {
    return DURATION_UNITS.MINUTES;
  }
  return DURATION_UNITS.HOURS;
}

/** Convertit une durée en minutes vers la valeur affichée dans l'unité choisie. */
export function minutesToDisplayValue(minutes, unit) {
  const value = Number(minutes);
  if (!Number.isFinite(value)) return '';
  if (unit === DURATION_UNITS.HOURS) {
    return value % 60 === 0 ? value / 60 : Number((value / 60).toFixed(1));
  }
  return value;
}

/** Convertit la saisie utilisateur (valeur + unité) en minutes entières. */
export function displayValueToMinutes(rawValue, unit) {
  const num = parseFloat(String(rawValue).replace(',', '.'));
  if (!Number.isFinite(num) || num <= 0) return null;
  if (unit === DURATION_UNITS.HOURS) {
    return Math.round(num * 60);
  }
  return Math.round(num);
}

/** Libellé lisible pour un résumé (badges, tooltips). */
export function formatDurationLabel(minutes) {
  const value = Number(minutes);
  if (!Number.isFinite(value) || value <= 0) return '—';

  if (value < 60) {
    return value === 1 ? '1 minute' : `${value} minutes`;
  }

  const wholeHours = Math.floor(value / 60);
  const remainder = value % 60;

  if (remainder === 0) {
    return wholeHours === 1 ? '1 heure' : `${wholeHours} heures`;
  }

  const hourLabel = wholeHours === 1 ? '1 h' : `${wholeHours} h`;
  const minuteLabel = remainder === 1 ? '1 min' : `${remainder} min`;
  return `${hourLabel} ${minuteLabel}`;
}

/** Borne min/max affichée selon l'unité active. */
export function getDurationBounds(minMinutes, maxMinutes, unit) {
  if (unit === DURATION_UNITS.HOURS) {
    return {
      min: minMinutes / 60,
      max: maxMinutes / 60,
      step: 0.5,
    };
  }
  return {
    min: minMinutes,
    max: maxMinutes,
    step: 1,
  };
}

/** Texte d'aide décrivant la plage autorisée dans l'unité active. */
export function formatDurationRangeHint(minMinutes, maxMinutes, unit) {
  if (unit === DURATION_UNITS.HOURS) {
    const formatHourBound = (minutes) => {
      const hours = minutes / 60;
      if (hours < 1) {
        return formatDurationLabel(minutes);
      }
      return Number.isInteger(hours)
        ? `${hours} h`
        : `${hours.toFixed(1).replace(/\.0$/, '')} h`;
    };
    return `${formatHourBound(minMinutes)} à ${formatHourBound(maxMinutes)}`;
  }
  return `${minMinutes} à ${maxMinutes.toLocaleString('fr-CH')} min`;
}
