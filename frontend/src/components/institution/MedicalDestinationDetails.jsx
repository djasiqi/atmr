import React from 'react';
import {
  DESTINATION_TYPE_MEDICAL,
  DESTINATION_TYPE_OTHER,
  MEDICAL_DESTINATION_OR_ERROR,
  MEDICAL_DESTINATION_OR_HINT,
  isMedicalDestinationType,
} from '../../utils/institutionDestinationDetails';
import styles from './MedicalDestinationDetails.module.css';

/**
 * Type de destination (Médical / Autre lieu) + détails.
 * Service OU Médecin uniquement si Médical.
 */
const MedicalDestinationDetails = ({
  establishmentId,
  serviceId,
  doctorId,
  establishment,
  service,
  doctor,
  onEstablishmentChange,
  onServiceChange,
  onDoctorChange,
  destinationType = DESTINATION_TYPE_MEDICAL,
  onDestinationTypeChange,
  showKindToggle = true,
  showError = false,
  compact = false,
  requireServiceOrDoctor,
}) => {
  const isMedical = isMedicalDestinationType(destinationType);
  const enforceMedical = requireServiceOrDoctor !== false && isMedical;
  const errorId = `${serviceId}-or-doctor-error`;
  const hintId = `${serviceId}-or-doctor-hint`;
  const describedBy = enforceMedical
    ? (showError ? errorId : hintId)
    : undefined;

  return (
    <div className={`${styles.block} ${compact ? styles.blockCompact : ''}`}>
      {showKindToggle && typeof onDestinationTypeChange === 'function' && (
        <div className={styles.kindRow}>
          <span className={styles.kindLabel}>Destination</span>
          <div className={styles.kindToggle} role="group" aria-label="Type de destination">
            <button
              type="button"
              className={`${styles.kindBtn} ${isMedical ? styles.kindBtnActive : ''}`}
              aria-pressed={isMedical}
              onClick={() => onDestinationTypeChange(DESTINATION_TYPE_MEDICAL)}
            >
              Médical
            </button>
            <button
              type="button"
              className={`${styles.kindBtn} ${!isMedical ? styles.kindBtnActive : ''}`}
              aria-pressed={!isMedical}
              onClick={() => onDestinationTypeChange(DESTINATION_TYPE_OTHER)}
            >
              Autre lieu
            </button>
          </div>
        </div>
      )}

      <div className={styles.group}>
        <label htmlFor={establishmentId} className={styles.label}>
          {isMedical ? 'Établissement / Lieu' : 'Lieu'}
        </label>
        <input
          type="text"
          id={establishmentId}
          value={establishment || ''}
          onChange={onEstablishmentChange}
          placeholder={isMedical
            ? 'Ex : HUG, Clinique des Grangettes'
            : 'Ex : Restaurant, gare, hôtel'}
          className={styles.input}
          autoComplete="off"
        />
      </div>

      {isMedical && (
        <div className={styles.orGroup} role="group" aria-describedby={describedBy}>
          <div className={styles.group}>
            <label htmlFor={serviceId} className={styles.label}>
              Service
              {enforceMedical && (
                <span className={styles.requiredMark} aria-hidden="true"> *</span>
              )}
            </label>
            <input
              type="text"
              id={serviceId}
              value={service || ''}
              onChange={onServiceChange}
              placeholder="Ex : Radiologie, Urgences, Cardiologie"
              className={`${styles.input} ${enforceMedical && showError ? styles.inputInvalid : ''}`}
              autoComplete="off"
              aria-invalid={(enforceMedical && showError) || undefined}
              aria-describedby={describedBy}
            />
          </div>

          <div className={styles.group}>
            <label htmlFor={doctorId} className={styles.label}>
              Médecin
              {enforceMedical && (
                <span className={styles.requiredMark} aria-hidden="true"> *</span>
              )}
            </label>
            <input
              type="text"
              id={doctorId}
              value={doctor || ''}
              onChange={onDoctorChange}
              placeholder="Ex : Dr Martin, Prof. Dupont"
              className={`${styles.input} ${enforceMedical && showError ? styles.inputInvalid : ''}`}
              autoComplete="off"
              aria-invalid={(enforceMedical && showError) || undefined}
              aria-describedby={describedBy}
            />
          </div>

          {enforceMedical && showError && (
            <p id={errorId} className={styles.error} role="alert">
              {MEDICAL_DESTINATION_OR_ERROR}
            </p>
          )}
          {enforceMedical && !showError && (
            <p id={hintId} className={styles.hint}>
              * {MEDICAL_DESTINATION_OR_HINT}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default MedicalDestinationDetails;
