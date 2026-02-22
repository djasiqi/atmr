// src/pages/company/Dashboard/components/InstitutionOffersTable.jsx
import React, { useState } from 'react';
import { FiCheckCircle, FiXCircle, FiClock, FiInbox } from 'react-icons/fi';
import styles from './ReservationTable.module.css';

/**
 * Formate une date ISO en "DD.MM.YYYY" et "HH:MM"
 */
const formatDateTime = (isoString) => {
  if (!isoString) return { date: '—', time: '' };
  const d = new Date(isoString);
  if (isNaN(d.getTime())) return { date: '—', time: '' };
  const pad = (n) => String(n).padStart(2, '0');
  return {
    date: `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`,
    time: `${pad(d.getHours())}:${pad(d.getMinutes())}`,
  };
};

/**
 * Convertit une date ISO en valeur pour input datetime-local (YYYY-MM-DDTHH:MM)
 */
const toDatetimeLocal = (isoString) => {
  if (!isoString) return '';
  const d = new Date(isoString);
  if (isNaN(d.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

/**
 * Modale inline pour proposer un horaire de prise en charge
 */
const ProposeTimeModal = ({ offer, onConfirm, onClose }) => {
  const req = offer?.transport_request || {};
  const [proposedTime, setProposedTime] = useState(
    toDatetimeLocal(req.scheduled_time) || ''
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!proposedTime) return;
    // Envoyer l'heure locale telle quelle (pas de conversion UTC)
    // datetime-local donne "2026-02-12T08:15" → on envoie tel quel
    onConfirm(offer.id, proposedTime);
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0,0,0,0.4)',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'white',
          borderRadius: 'var(--radius-lg)',
          padding: '24px',
          minWidth: 380,
          maxWidth: 440,
          boxShadow: 'var(--shadow-lg)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 style={{ margin: '0 0 16px', fontSize: '16px' }}>
          Proposer un horaire de prise en charge
        </h3>

        {/* Résumé de la demande */}
        <div
          style={{
            background: 'var(--bg-secondary)',
            borderRadius: 'var(--radius-md)',
            padding: '12px',
            marginBottom: '16px',
            fontSize: '13px',
          }}
        >
          <div>
            <strong>{req.institution_name || 'Institution'}</strong>
          </div>
          <div style={{ marginTop: 4 }}>
            Horaire demandé :{' '}
            <strong>
              {formatDateTime(req.scheduled_time).date}{' '}
              {formatDateTime(req.scheduled_time).time}
            </strong>
          </div>
          <div style={{ marginTop: 4, color: 'var(--text-tertiary)' }}>
            {req.pickup_location} → {req.dropoff_location}
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <label
            style={{
              display: 'block',
              marginBottom: '8px',
              fontWeight: 600,
              fontSize: '13px',
            }}
          >
            Horaire proposé
          </label>
          <input
            type="datetime-local"
            value={proposedTime}
            onChange={(e) => setProposedTime(e.target.value)}
            required
            style={{
              width: '100%',
              padding: '10px 12px',
              border: '1px solid var(--border-primary)',
              borderRadius: 'var(--radius-md)',
              fontSize: '14px',
              marginBottom: '16px',
              boxSizing: 'border-box',
            }}
          />

          <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: '8px 16px',
                border: '1px solid var(--border-primary)',
                borderRadius: 'var(--radius-md)',
                background: 'white',
                cursor: 'pointer',
                fontSize: '13px',
              }}
            >
              Annuler
            </button>
            <button
              type="submit"
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: 'var(--radius-md)',
                background: 'var(--brand-primary)',
                color: 'white',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '13px',
              }}
            >
              Accepter avec cet horaire
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

/**
 * Tableau des offres de transport institutionnelles.
 * Affiche les demandes reçues d'institutions avec possibilité d'accepter/proposer/refuser.
 */
const InstitutionOffersTable = ({ offers = [], loading, onAccept, onReject }) => {
  const [proposeOffer, setProposeOffer] = useState(null);

  if (loading) {
    return (
      <div className={styles.emptyState}>
        <FiInbox className={styles.emptyIcon} size={40} />
        <p className={styles.emptyTitle}>Chargement...</p>
        <p className={styles.emptySubtitle}>Récupération des demandes institutions</p>
      </div>
    );
  }

  if (!offers.length) {
    return (
      <div className={styles.emptyState}>
        <FiInbox className={styles.emptyIcon} size={40} />
        <p className={styles.emptyTitle}>Aucune demande d'institution en attente</p>
        <p className={styles.emptySubtitle}>Les nouvelles demandes apparaîtront ici automatiquement</p>
      </div>
    );
  }

  return (
    <>
      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Institution</th>
              <th>Date / Heure</th>
              <th>Trajet</th>
              <th>Type</th>
              <th>Statut</th>
              <th className={styles.actionsCell}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {offers.map((offer) => {
              const req = offer.transport_request || {};
              const { date, time } = formatDateTime(req.scheduled_time);
              const mobility = req.mobility;
              const mobilityTags = [];
              if (mobility) {
                if (mobility.wheelchair) mobilityTags.push('Fauteuil');
                if (mobility.stretcher) mobilityTags.push('Brancard');
                if (mobility.oxygen) mobilityTags.push('O₂');
                if (mobility.walking_aid) mobilityTags.push('Aide marche');
              }

              return (
                <tr key={offer.id} className={styles.tableRow}>
                  <td className={styles.clientCell}>
                    <strong>{req.institution_name || 'Institution'}</strong>
                    {req.patient_name && (
                      <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
                        {req.patient_name}
                      </div>
                    )}
                  </td>
                  <td>
                    <div>{date}</div>
                    <div style={{ fontWeight: 600 }}>{time}</div>
                    {req.scheduled_time_type && (
                      <div style={{ fontSize: '10px', marginTop: 2, color: req.scheduled_time_type === 'arrival' ? 'var(--brand-primary)' : 'var(--text-tertiary)' }}>
                        {req.scheduled_time_type === 'arrival' ? 'Heure RDV' : 'Heure départ'}
                      </div>
                    )}
                  </td>
                  <td className={styles.locationCell}>
                    <div>
                      <strong>De:</strong> {req.pickup_location || '—'}
                    </div>
                    <div>
                      <strong>À:</strong> {req.dropoff_location || '—'}
                    </div>
                    {req.is_round_trip && (
                      <div
                        style={{
                          fontSize: '11px',
                          color: 'var(--brand-primary)',
                          marginTop: 2,
                        }}
                      >
                        ↩ Aller-retour
                      </div>
                    )}
                  </td>
                  <td>
                    <div style={{ fontSize: '13px' }}>
                      {req.mission_type === 'patient_transport'
                        ? 'Patient'
                        : req.mission_type || '—'}
                    </div>
                    {mobilityTags.length > 0 && (
                      <div
                        style={{
                          display: 'flex',
                          gap: 4,
                          flexWrap: 'wrap',
                          marginTop: 4,
                        }}
                      >
                        {mobilityTags.map((tag) => (
                          <span
                            key={tag}
                            style={{
                              fontSize: '10px',
                              padding: '1px 6px',
                              borderRadius: 'var(--radius-full)',
                              background: 'var(--info-bg)',
                              color: 'var(--info-primary)',
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td>
                    <span className={`${styles.statusBadge} ${styles.pending}`}>
                      {offer.can_respond ? 'En attente' : 'Expiré'}
                    </span>
                    {offer.expires_at && (
                      <div
                        style={{
                          fontSize: '10px',
                          color: 'var(--text-tertiary)',
                          marginTop: 2,
                        }}
                      >
                        Exp: {formatDateTime(offer.expires_at).date}{' '}
                        {formatDateTime(offer.expires_at).time}
                      </div>
                    )}
                  </td>
                  <td className={styles.actionsCell}>
                    {offer.can_respond ? (
                      <div style={{ display: 'flex', gap: '2px', alignItems: 'center' }}>
                        {/* Accepter (avec l'horaire demandé) */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onAccept?.(offer.id);
                          }}
                          title="Accepter (horaire demandé)"
                          className={`${styles.actionButton} ${styles.acceptButton}`}
                        >
                          <FiCheckCircle size={18} />
                        </button>

                        {/* Proposer un horaire */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setProposeOffer(offer);
                          }}
                          title="Accepter avec un horaire différent"
                          className={styles.actionButton}
                          style={{ color: 'var(--brand-primary)' }}
                        >
                          <FiClock size={18} />
                        </button>

                        {/* Refuser */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onReject?.(offer.id);
                          }}
                          title="Refuser la demande"
                          className={`${styles.actionButton} ${styles.rejectButton}`}
                        >
                          <FiXCircle size={18} />
                        </button>
                      </div>
                    ) : (
                      <span style={{ color: 'var(--text-tertiary)', fontSize: '12px' }}>
                        Aucune action
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Modale proposition d'horaire */}
      {proposeOffer && (
        <ProposeTimeModal
          offer={proposeOffer}
          onConfirm={(offerId, isoTime) => {
            onAccept?.(offerId, isoTime);
            setProposeOffer(null);
          }}
          onClose={() => setProposeOffer(null)}
        />
      )}
    </>
  );
};

export default InstitutionOffersTable;
