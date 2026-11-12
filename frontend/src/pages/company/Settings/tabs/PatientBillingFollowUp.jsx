import React, { useCallback, useEffect, useState } from 'react';
import apiClient from '../../../../utils/apiClient';
import { dossierBusinessState } from '../../billing/financialState';
import styles from './CashRegisterFollowUp.module.css';

const REVIEW_LABEL = {
  absent: 'Non renseigné',
  pending_review: 'À vérifier',
  documented_scope: 'Vérifié pour le périmètre documenté',
  refused: 'Refusé',
  insufficient_proof: 'Justificatif insuffisant',
};

/**
 * Suivi administratif des dossiers patient.
 * Le responsable ne fabrique pas l'acceptation du patient.
 */
export default function PatientBillingFollowUp() {
  const [payload, setPayload] = useState(null);
  const [hidden, setHidden] = useState(false);
  const [sessionExpired, setSessionExpired] = useState(false);
  const [notice, setNotice] = useState('');
  const [openId, setOpenId] = useState(null);

  const load = useCallback(() => {
    apiClient
      .get('/companies/me/patient-billing-dossiers')
      .then((response) => {
        setPayload(response.data);
        setHidden(false);
        setSessionExpired(false);
      })
      .catch((error) => {
        if (error?.response?.status === 401) {
          setSessionExpired(true);
          setPayload(null);
          return;
        }
        setHidden(true);
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (sessionExpired) {
    return (
      <section aria-label="Dossiers de facturation patient" className={styles.wrap}>
        <h3>Dossiers de facturation patient</h3>
        <p>Session expirée</p>
      </section>
    );
  }

  if (hidden || !payload) return null;

  function review(id, result) {
    apiClient
      .post(`/companies/me/patient-billing-dossiers/${id}/review`, {
        result,
        idempotency_key: `revue-${id}-${result}`,
        powers_verified: false,
      })
      .then(() => {
        setNotice('Note administrative enregistrée. Ce n’est pas une acceptation du patient.');
        load();
      })
      .catch(() => {
        setNotice('La revue n’a pas été enregistrée.');
      });
  }

  return (
    <section aria-label="Dossiers de facturation patient" className={styles.wrap}>
      <h3>Dossiers de facturation patient</h3>
      {notice ? <p>{notice}</p> : null}
      <ul className={styles.list}>
        {(payload.cases || []).map((row) => (
          <li key={row.id}>
            <p>{dossierBusinessState(row)}</p>
            <p>{`Réservation ${row.booking_id} — ${row.company_name || ''}`}</p>
            <p className={styles.amount}>{`${row.amount_chf} CHF`}</p>
            <p>{`Débiteur : ${row.debtor_label}`}</p>
            <p>{`Fondement : ${row.contractual_basis}`}</p>
            <p>{`Représentant : ${row.representative_name || 'aucun'}`}</p>
            <p>{`Pouvoirs : ${REVIEW_LABEL[row.representative_review] || row.representative_review}`}</p>
            <p>{row.draft_invoice_id ? `Brouillon ${row.draft_invoice_id}` : 'Pas de brouillon'}</p>
            <button type="button" onClick={() => setOpenId(openId === row.id ? null : row.id)}>
              Historique
            </button>
            {openId === row.id
              ? (row.history || []).map((event) => (
                  <p key={`${row.id}-${event.action}`}>{`${event.action} — ${event.to_state}`}</p>
                ))
              : null}
            {row.representative_name ? (
              <div className={styles.actions}>
                <button type="button" onClick={() => review(row.id, 'documented_scope')}>
                  Noter le périmètre documenté
                </button>
                <button type="button" onClick={() => review(row.id, 'insufficient_proof')}>
                  Justificatif insuffisant
                </button>
                <button type="button" onClick={() => review(row.id, 'refused')}>
                  Refuser le pouvoir annoncé
                </button>
              </div>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}
