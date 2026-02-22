// src/pages/Dashboard/ReservationDetailsModal.jsx
// ✅ P3: Ajout de l'affichage des bons de transport liés à une course
import React, { useCallback, useEffect, useState, useMemo, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import styles from '../CompanyDashboard.module.css';
import { renderBookingDateTime } from '../../../../utils/formatDate';
import { fetchTransportVouchers } from '../../../../services/transportVoucherService';

const STATUS_LABELS = {
  draft: 'Brouillon',
  submitted: 'Soumis',
  validated: 'Validé',
  rejected: 'Rejeté',
  expired: 'Expiré',
};

const STATUS_COLORS = {
  draft: '#6b7280',
  submitted: '#3b82f6',
  validated: '#10b981',
  rejected: '#ef4444',
  expired: '#f59e0b',
};

const TYPE_LABELS = {
  clinic: 'Clinique',
  insurance: 'Assurance',
  other: 'Autre',
};

const ReservationDetailsModal = ({ reservation, onClose }) => {
  const [vouchers, setVouchers] = useState([]);
  const [loadingVouchers, setLoadingVouchers] = useState(false);
  const [searchParams] = useSearchParams();
  const lastVoucherErrorRef = useRef(null);
  const formatCurrency = useCallback((value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '-';
    return `${numeric.toFixed(2)} CHF`;
  }, []);
  const originalAmount =
    reservation?.amount_original
    ?? reservation?.original_amount
    ?? reservation?.requested_amount;
  const adjustedDelta = Number.isFinite(Number(originalAmount))
    ? Number(reservation?.amount ?? 0) - Number(originalAmount)
    : null;
  const returnTo = useMemo(() => {
    const raw = searchParams.get('returnTo');
    if (!raw) return null;
    try {
      const decoded = decodeURIComponent(raw);
      if (
        decoded.startsWith('/dashboard/company/')
        || decoded.startsWith('/company/')
      ) {
        return decoded;
      }
      return null;
    } catch {
      return null;
    }
  }, [searchParams]);

  const loadVouchers = useCallback(async () => {
    if (!reservation?.id) return;
    
    try {
      setLoadingVouchers(true);
      const response = await fetchTransportVouchers({ booking_id: reservation.id });
      setVouchers(response?.data || []);
    } catch (e) {
      const errorKey = `${reservation.id}:${e?.response?.status || e?.message || 'unknown'}`;
      if (lastVoucherErrorRef.current !== errorKey) {
        lastVoucherErrorRef.current = errorKey;
        console.error('[ReservationDetailsModal] Error loading vouchers:', e);
      }
    } finally {
      setLoadingVouchers(false);
    }
  }, [reservation?.id]);

  useEffect(() => {
    if (reservation?.id) {
      loadVouchers();
    }
  }, [reservation?.id, loadVouchers]);

  return (
    <div className={styles.modal}>
      <div className={styles.modalContent}>
        <h3>Détails de la réservation #{reservation.id}</h3>
        <p>
          <strong>Client :</strong> {reservation.client_name}
        </p>
        {reservation.client?.birth_date && (
          <p>
            <strong>Date de naissance :</strong>{' '}
            {new Date(reservation.client.birth_date).toLocaleDateString('fr-FR', {
              day: '2-digit',
              month: '2-digit',
              year: 'numeric',
            })}
          </p>
        )}
        <p>
          <strong>Date / Heure :</strong> {renderBookingDateTime(reservation)}
        </p>
        <p>
          <strong>Montant facturé :</strong> {formatCurrency(reservation?.amount)}
        </p>
        {Number.isFinite(Number(originalAmount)) && Number.isFinite(Number(adjustedDelta)) && (
          <p>
            <strong>Montant saisi :</strong> {formatCurrency(originalAmount)}
            {Math.abs(adjustedDelta) >= 0.01 && (
              <span>
                {' '}
                — Ajusté : {adjustedDelta >= 0 ? '+' : '-'}
                {formatCurrency(Math.abs(adjustedDelta))}
              </span>
            )}
          </p>
        )}
        <p>
          <strong>Statut :</strong> {reservation.status}
        </p>
        {reservation.phone && (
          <p>
            <strong>Téléphone :</strong> {reservation.phone}
          </p>
        )}
        {reservation.pickup_location && (
          <p>
            <strong>Départ :</strong> {reservation.pickup_location}
          </p>
        )}
        {reservation.dropoff_location && (
          <p>
            <strong>Arrivée :</strong> {reservation.dropoff_location}
          </p>
        )}
        {reservation.instructions && (
          <p>
            <strong>Instructions :</strong> {reservation.instructions}
          </p>
        )}

        {/* ✅ P0.5: Section facturation — résolution payeur */}
        <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid #e5e7eb' }}>
          <h4 style={{ marginBottom: '0.5rem' }}>💰 Facturation</h4>
          {(() => {
            const meta = reservation.metadata_json || {};
            const billingStatus = meta.billing_resolution_status;
            const billingSource = meta.billing_resolution_source;
            const billingIntent = meta.billing_resolution_intent || reservation.billed_to_type;
            const isFailed = billingStatus && billingStatus.startsWith('failed');
            const isInstitutionBooking = !!meta.institution_id;

            // Labels
            const intentLabels = {
              institution: 'Institution',
              clinic: 'Clinique',
              patient: 'Patient',
              curator: 'Curateur',
              spc: 'SPC',
              other: 'Autre',
            };
            const statusLabels = {
              success: 'Payeur résolu',
              failed_missing_institution_address: 'Adresse institution manquante',
              failed_missing_payer_info: 'Infos payeur manquantes',
              failed_missing_patient_info: 'Infos patient incomplètes',
              failed_error: 'Erreur de résolution',
            };

            // Si ce n'est pas un booking institution, afficher simplement le type
            if (!isInstitutionBooking) {
              return reservation.billed_to_type ? (
                <p>
                  <strong>Facturé à :</strong>{' '}
                  {intentLabels[reservation.billed_to_type] || reservation.billed_to_type}
                </p>
              ) : (
                <p style={{ color: '#6b7280', fontSize: '0.875rem' }}>
                  Informations de facturation standard.
                </p>
              );
            }

            return (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <strong>Facturé à :</strong>
                  <span>{intentLabels[billingIntent] || billingIntent || '—'}</span>
                  {billingStatus && (
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        background: isFailed ? '#fef2f2' : '#ecfdf5',
                        color: isFailed ? '#dc2626' : '#059669',
                        border: `1px solid ${isFailed ? '#fecaca' : '#a7f3d0'}`,
                      }}
                    >
                      {isFailed ? '⚠️' : '✅'} {statusLabels[billingStatus] || billingStatus}
                    </span>
                  )}
                </div>

                {meta.institution_name && (
                  <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                    Institution : {meta.institution_name}
                  </p>
                )}

                {isFailed && (
                  <div
                    style={{
                      marginTop: '0.5rem',
                      padding: '0.75rem',
                      background: '#fffbeb',
                      border: '1px solid #fcd34d',
                      borderRadius: '6px',
                      fontSize: '0.85rem',
                      color: '#92400e',
                    }}
                  >
                    <strong>Action requise :</strong>{' '}
                    {billingStatus === 'failed_missing_institution_address'
                      ? `L'institution${meta.institution_name ? ` (${meta.institution_name})` : ''} n'a pas d'adresse de facturation configurée. Contactez l'institution pour qu'elle renseigne son adresse dans Paramètres > Facturation, ou complétez manuellement le destinataire.`
                      : billingStatus === 'failed_missing_payer_info'
                        ? `Les coordonnées du tiers payeur (nom et adresse) n'ont pas été transmises par l'institution${meta.institution_name ? ` (${meta.institution_name})` : ''}. Contactez l'institution pour obtenir ces informations avant de facturer.`
                        : billingStatus === 'failed_missing_patient_info'
                          ? "Les informations du patient (nom ou adresse) sont incomplètes pour la facturation. Vérifiez le dossier client ou contactez l'institution pour obtenir l'adresse de domicile."
                          : "Une erreur technique est survenue lors de la résolution automatique du destinataire. Veuillez saisir manuellement les informations de facturation."
                    }
                  </div>
                )}

                {billingSource && !isFailed && (
                  <p style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                    Source : {billingSource.replace(/_/g, ' ')}
                  </p>
                )}
              </div>
            );
          })()}
        </div>

        {/* ✅ P3: Bons de transport liés */}
        <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid #e5e7eb' }}>
          <h4 style={{ marginBottom: '0.5rem' }}>🎫 Bons de transport</h4>
          {loadingVouchers ? (
            <p style={{ color: '#6b7280' }}>Chargement...</p>
          ) : vouchers.length === 0 ? (
            <p style={{ color: '#6b7280', fontSize: '0.875rem' }}>Aucun bon de transport lié à cette course.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {vouchers.map((v) => (
                <div
                  key={v.id}
                  style={{
                    padding: '0.75rem',
                    background: '#f9fafb',
                    borderRadius: '6px',
                    border: '1px solid #e5e7eb',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                    <div>
                      <strong>Bon #{v.id}</strong> - {TYPE_LABELS[v.type] || v.type}
                    </div>
                    <span
                      style={{
                        padding: '0.25rem 0.5rem',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        backgroundColor: `${STATUS_COLORS[v.status]}20`,
                        color: STATUS_COLORS[v.status],
                        fontWeight: '500',
                      }}
                    >
                      {STATUS_LABELS[v.status] || v.status}
                    </span>
                  </div>
                  {v.external_ref && (
                    <div style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.25rem' }}>
                      <strong>Réf:</strong> {v.external_ref}
                    </div>
                  )}
                  {v.valid_from && (
                    <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                      <strong>Période:</strong>{' '}
                      {v.valid_to
                        ? `${new Date(v.valid_from).toLocaleDateString()} - ${new Date(v.valid_to).toLocaleDateString()}`
                        : `À partir du ${new Date(v.valid_from).toLocaleDateString()}`}
                    </div>
                  )}
                  {v.files && v.files.length > 0 && (
                    <div style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                      <strong>Fichiers:</strong>{' '}
                      {v.files.map((f, idx) => (
                        <React.Fragment key={f.id}>
                          <a
                            href={f.file_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ color: '#3b82f6', marginLeft: '0.25rem' }}
                          >
                            📎 {f.filename}
                          </a>
                          {idx < v.files.length - 1 && ', '}
                        </React.Fragment>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {returnTo && (
          <button
            className={styles.cancelButton}
            onClick={() => window.location.assign(returnTo)}
          >
            ↩ Retour au contrôle facturation
          </button>
        )}
        <button className={styles.cancelButton} onClick={onClose}>
          Fermer
        </button>
      </div>
    </div>
  );
};

export default ReservationDetailsModal;
