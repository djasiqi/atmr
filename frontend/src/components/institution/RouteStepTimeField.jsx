import React, { forwardRef } from 'react';
import InlineTimePicker from '../ui/InlineTimePicker';
import styles from '../../pages/institution/Requests/InstitutionRequestForm.module.css';

/**
 * Heure d'une étape du parcours.
 * Règle : heure saisie ⇒ confirmée ; heure vide ⇒ non confirmée.
 * La confirmation est dérivée automatiquement de la présence d'une heure.
 */
const RouteStepTimeField = forwardRef(function RouteStepTimeField({
  timeValue = '',
  onTimeChange,
  onConfirmedChange,
  inputId,
  disabled = false,
  label,
}, ref) {
  const handleTimeChange = (value) => {
    onTimeChange?.(value);
    onConfirmedChange?.(Boolean(value?.trim()));
  };

  return (
    <div className={styles.routeStepTimeBlock}>
      <div className={styles.routeStepTimeRow}>
        <InlineTimePicker
          ref={ref}
          inputId={inputId}
          value={timeValue}
          onChange={handleTimeChange}
          placeholder="Heure"
          disabled={disabled}
          title={label}
          ariaLabel={label}
        />
      </div>
    </div>
  );
});

export default RouteStepTimeField;
