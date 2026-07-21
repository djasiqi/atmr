import React from 'react';
import ChipSelect from '../ui/ChipSelect';
import styles from './DestinationBillingOverride.module.css';

export const DESTINATION_BILLING_OPTIONS = [
  { value: 'patient', label: 'Patient' },
  { value: 'institution', label: 'Institution' },
];

/**
 * Override facturation par destination (hérite du payeur principal par défaut).
 */
const DestinationBillingOverride = ({
  idPrefix = 'dest-billing',
  useCustomBilling = false,
  billingOverride = 'patient',
  onUseCustomBillingChange,
  onBillingOverrideChange,
  disabled = false,
  compact = false,
}) => {
  const checkboxId = `${idPrefix}-custom`;
  const selectId = `${idPrefix}-intent`;
  const resolvedOverride = ['patient', 'institution'].includes(billingOverride)
    ? billingOverride
    : 'patient';

  return (
    <div className={`${styles.wrap} ${compact ? styles.wrapCompact : ''}`}>
      <label
        htmlFor={checkboxId}
        className={`${styles.toggle} ${disabled ? styles.toggleDisabled : ''}`}
      >
        <input
          id={checkboxId}
          type="checkbox"
          className={styles.checkbox}
          checked={Boolean(useCustomBilling)}
          disabled={disabled}
          onChange={(e) => onUseCustomBillingChange?.(e.target.checked)}
        />
        <span className={styles.label}>Facturation spécifique</span>
      </label>

      {useCustomBilling && (
        <div className={styles.field}>
          <ChipSelect
            id={selectId}
            options={DESTINATION_BILLING_OPTIONS}
            value={resolvedOverride}
            onChange={onBillingOverrideChange}
            placeholder="Facturer à"
            disabled={disabled}
          />
        </div>
      )}
    </div>
  );
};

export default DestinationBillingOverride;
