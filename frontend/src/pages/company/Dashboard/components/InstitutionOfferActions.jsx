import React from 'react';
import { FiCheckCircle, FiXCircle, FiClock, FiZap } from 'react-icons/fi';
import { resolveInstitutionOfferActions } from '../../../../utils/institutionOfferActions';
import styles from './ReservationTable.module.css';

/**
 * Matrice d'actions offres institutionnelles — partagée table / cartes mobiles / ReservationTable.
 */
export default function InstitutionOfferActions({
  offer,
  onValidate,
  onAcceptNow,
  onPlan,
  onReject,
  className,
}) {
  const actions = resolveInstitutionOfferActions(offer);
  if (!actions.canRespond) return null;

  return (
    <div className={className || styles.institutionOfferActions}>
      {actions.canValidate ? (
        <button
          type="button"
          onClick={onValidate}
          title={actions.validateLabel}
          aria-label={actions.validateLabel}
          className={`${styles.actionButton} ${styles.acceptButton} ${styles.touchTarget}`}
        >
          <FiCheckCircle size={16} aria-hidden />
        </button>
      ) : null}
      {actions.canAcceptNow ? (
        <button
          type="button"
          onClick={onAcceptNow}
          title={actions.acceptNowLabel}
          aria-label={actions.acceptNowLabel}
          className={`${styles.actionButton} ${styles.touchTarget}`}
          style={{ color: 'var(--warning-primary, #ea580c)' }}
        >
          <FiZap size={16} aria-hidden />
        </button>
      ) : null}
      {actions.canPlan ? (
        <button
          type="button"
          onClick={onPlan}
          title={actions.planLabel}
          aria-label={actions.planLabel}
          className={`${styles.actionButton} ${styles.touchTarget}`}
          style={{ color: 'var(--brand-primary)' }}
        >
          <FiClock size={16} aria-hidden />
        </button>
      ) : null}
      {actions.canReject ? (
        <button
          type="button"
          onClick={onReject}
          title={actions.rejectLabel}
          aria-label={actions.rejectLabel}
          className={`${styles.actionButton} ${styles.rejectButton} ${styles.touchTarget}`}
        >
          <FiXCircle size={16} aria-hidden />
        </button>
      ) : null}
    </div>
  );
}
