import React from 'react';
import ChipSelect from '../ui/ChipSelect';

export const DESTINATION_BILLING_OPTIONS = [
  { value: 'patient', label: 'Patient' },
  { value: 'institution', label: 'Institution' },
  { value: 'insurance', label: 'Assurance' },
  { value: 'curator', label: 'Curateur' },
  { value: 'spc', label: 'SPC' },
  { value: 'other', label: 'Autre' },
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
  return (
    <div style={{ marginTop: compact ? 4 : 8 }}>
      <label
        htmlFor={checkboxId}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: '0.82rem',
          color: 'var(--text-muted, #64748B)',
          cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        <input
          id={checkboxId}
          type="checkbox"
          checked={Boolean(useCustomBilling)}
          disabled={disabled}
          onChange={(e) => onUseCustomBillingChange?.(e.target.checked)}
        />
        Facturation spécifique pour cette destination
      </label>
      {useCustomBilling && (
        <div style={{ marginTop: 6, maxWidth: 280 }}>
          <ChipSelect
            id={`${idPrefix}-intent`}
            options={DESTINATION_BILLING_OPTIONS}
            value={billingOverride || 'patient'}
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
