import React, { useCallback, useEffect, useState } from 'react';
import apiClient from '../../../../utils/apiClient';
import styles from './CashRegisterFollowUp.module.css';

const STATUS_LABELS = {
  a_traiter: 'À traiter',
  en_cours_examen: 'En cours d’examen',
  regularise: 'Régularisé',
  cloture_justifie: 'Clôturé avec justification',
};

/**
 * File administrative des signalements de non-règlement.
 * Transport terminé ≠ règlement effectué. Aucune signature inventée.
 */
export default function SettlementIncidentsFollowUp() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [openId, setOpenId] = useState(null);
  const [justifications, setJustifications] = useState({});
  const [notes, setNotes] = useState({});
  const [notice, setNotice] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    apiClient
      .get('/companies/me/settlement-incidents')
      .then((response) => {
        setCases(response.data?.cases || []);
        setFailed(false);
      })
      .catch(() => {
        setCases([]);
        setFailed(true);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const setStatus = async (incidentId, status) => {
    const justification = (justifications[incidentId] || '').trim();
    try {
      await apiClient.post(`/companies/me/settlement-incidents/${incidentId}/status`, {
        status,
        justification,
        idempotency_key: `ui-status-${incidentId}-${status}`,
      });
      setNotice('Statut mis à jour.');
      load();
    } catch (error) {
      setNotice(error?.response?.data?.error || 'Mise à jour refusée.');
    }
  };

  const notePayment = async (incidentId) => {
    const note = (notes[incidentId] || '').trim();
    try {
      await apiClient.post(
        `/companies/me/settlement-incidents/${incidentId}/later-payment`,
        {
          note,
          idempotency_key: `ui-later-${incidentId}`,
        }
      );
      setNotice('Paiement ultérieur noté (sans encaissement chauffeur fictif).');
      load();
    } catch (error) {
      setNotice(error?.response?.data?.error || 'Note refusée.');
    }
  };

  if (loading) return <p className={styles.wrap}>Chargement des signalements…</p>;
  if (failed) {
    return (
      <p className={styles.wrap}>
        Impossible de charger la file des signalements de règlement.
      </p>
    );
  }

  return (
    <div className={styles.wrap} data-testid="settlement-incidents-followup">
      <h2>Signalements de non-règlement</h2>
      <p>
        La mission peut être terminée sans règlement. Ce dossier n’est ni un paiement, ni
        une signature, ni une autorisation de facturation différée.
      </p>
      {notice ? <p role="status">{notice}</p> : null}
      {cases.length === 0 ? <p>Aucun signalement à traiter.</p> : null}
      <ul className={styles.list}>
        {cases.map((item) => (
          <li key={item.id}>
            <button type="button" onClick={() => setOpenId(openId === item.id ? null : item.id)}>
              Transport #{item.booking_id} — {item.debtor_label || 'Client'} —{' '}
              {item.amount_due_chf} {item.currency} — {STATUS_LABELS[item.status] || item.status}
            </button>
            {openId === item.id ? (
              <div className={styles.summary}>
                <p>Chauffeur #{item.driver_id}</p>
                <p>Date : {item.transport_date || '—'}</p>
                <p>Motif : {item.motif}</p>
                <p>Observations : {item.observation || '—'}</p>
                <p>Règlement enregistré : non</p>
                <p>Transport terminé / règlement en attente : oui</p>
                <label>
                  Justification
                  <input
                    value={justifications[item.id] || ''}
                    onChange={(event) =>
                      setJustifications({
                        ...justifications,
                        [item.id]: event.target.value,
                      })
                    }
                  />
                </label>
                <div className={styles.actions}>
                  <button type="button" onClick={() => setStatus(item.id, 'en_cours_examen')}>
                    En cours d’examen
                  </button>
                  <button type="button" onClick={() => setStatus(item.id, 'regularise')}>
                    Régulariser
                  </button>
                  <button type="button" onClick={() => setStatus(item.id, 'cloture_justifie')}>
                    Clôturer
                  </button>
                </div>
                <label>
                  Note paiement ultérieur
                  <input
                    value={notes[item.id] || ''}
                    onChange={(event) =>
                      setNotes({
                        ...notes,
                        [item.id]: event.target.value,
                      })
                    }
                  />
                </label>
                <button type="button" onClick={() => notePayment(item.id)}>
                  Noter un paiement ultérieur
                </button>
                <h3>Historique</h3>
                <ul>
                  {(item.history || []).map((event) => (
                    <li key={event.id}>
                      {event.at} — {event.actor_role} — {event.action} — {event.detail}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
