import React, { useCallback, useMemo, useState, useEffect } from 'react';
import apiClient from '../../../../utils/apiClient';
import { cashBusinessState } from '../../billing/financialState';
import {
  allocationLabel,
  cashActionError,
  cashControlCsv,
  cashHistoryLabel,
  filterCashCases,
  handoverLabel,
} from './cashRegisterView';
import styles from './CashRegisterFollowUp.module.css';

const EMPTY_FILTERS = {
  driver: '',
  status: '',
  operation: '',
  handover: '',
  discrepancy: '',
  from: '',
  to: '',
};

function money(value) {
  if (value == null || value === '') return 'Non communiqué';
  return `${value} CHF`;
}

function amountKnown(value) {
  return value != null && String(value).trim() !== '';
}

function correctionsOf(item) {
  return (item.operations || []).filter((line) => line.nature === 'correction');
}

function refundsOf(item) {
  return (item.operations || []).filter((line) => line.nature === 'remboursement');
}

function hasConstat(item, handoverId) {
  return (item.reconciliations || []).some(
    (line) => line.status === 'constate' && String(line.handover_id) === String(handoverId)
  );
}

/**
 * Règlements consulte. La caisse confirme et rapproche.
 * Les montants affichés sont ceux du registre, sans recalcul.
 * Un environnement sans ces fonctions n'affiche aucun montant inventé.
 */
export default function CashRegisterFollowUp({ focus = 'caisse', canManageCash = true }) {
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);
  const [failed, setFailed] = useState(false);
  const [forbidden, setForbidden] = useState(false);
  const [sessionExpired, setSessionExpired] = useState(false);
  const [notice, setNotice] = useState(null);
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [openId, setOpenId] = useState(null);
  const [counted, setCounted] = useState({});
  const [motifs, setMotifs] = useState({});
  const [specimen, setSpecimen] = useState(null);
  const [exportPreview, setExportPreview] = useState('');
  const cashDesk = focus === 'caisse';

  const load = useCallback(() => {
    setLoading(true);
    apiClient
      .get('/companies/me/cash-register')
      .then((response) => {
        setPayload(response.data);
        setUnavailable(false);
        setFailed(false);
        setForbidden(false);
        setSessionExpired(false);
      })
      .catch((error) => {
        const status = error?.response?.status;
        setPayload(null);
        setSessionExpired(status === 401);
        setUnavailable(status === 404);
        setForbidden(status === 403);
        setFailed(status !== 401 && status !== 404 && status !== 403);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const cases = useMemo(
    () => filterCashCases(payload?.cases, filters),
    [payload, filters]
  );

  const title = cashDesk ? 'Caisse chauffeurs' : 'Règlements';

  if (loading && !payload) {
    return (
      <section aria-label={title} className={styles.wrap}>
        <h3>{title}</h3>
        <p>Chargement des informations.</p>
      </section>
    );
  }

  if (sessionExpired) {
    return (
      <section aria-label="Registre de caisse" className={styles.wrap}>
        <h3>Registre de caisse</h3>
        <p>Session expirée</p>
      </section>
    );
  }

  if (forbidden) {
    return (
      <section aria-label={title} className={styles.wrap}>
        <h3>{title}</h3>
        <p>Accès refusé. Cette consultation est réservée au responsable de l’entreprise.</p>
      </section>
    );
  }

  if (unavailable) {
    return (
      <section aria-label={cashDesk ? 'Caisse chauffeurs' : 'Règlements'} className={styles.wrap}>
        <h3>{cashDesk ? 'Caisse chauffeurs' : 'Règlements'}</h3>
        <p>La caisse n’est pas disponible dans cet environnement. Aucun montant n’est affiché.</p>
      </section>
    );
  }

  if (failed) {
    return (
      <section aria-label={cashDesk ? 'Caisse chauffeurs' : 'Règlements'} className={styles.wrap}>
        <h3>{cashDesk ? 'Caisse chauffeurs' : 'Règlements'}</h3>
        <p>Les informations de caisse n’ont pas pu être chargées.</p>
      </section>
    );
  }

  if (!payload) return null;

  function setFilter(name, value) {
    setFilters((current) => ({ ...current, [name]: value }));
  }

  function confirm(collectionId, handover) {
    const url = `/companies/me/cash-register/${collectionId}/confirm-handover`;
    const request = handover?.idempotency_key
      ? apiClient.post(url, { idempotency_key: handover.idempotency_key })
      : apiClient.post(url);
    request
      .then(() => {
        setNotice('Remise confirmée. L’écart de caisse reste un contrôle distinct.');
        load();
      })
      .catch((error) => setNotice(cashActionError(error)));
  }

  function reconcile(collectionId, handoverId) {
    const key = handoverId || collectionId;
    apiClient
      .post(`/companies/me/cash-register/${collectionId}/reconcile`, {
        counted_chf: counted[key] || '',
        reason: motifs[key] || 'rapprochement',
        idempotency_key: `ecart-${key}`,
        ...(handoverId ? { handover_id: handoverId } : {}),
      })
      .then(() => {
        setNotice('Écart enregistré.');
        setCounted((current) => ({ ...current, [key]: '' }));
        load();
      })
      .catch((error) => setNotice(cashActionError(error)));
  }

  function receipt(id) {
    apiClient
      .get(`/companies/me/cash-register/${id}/receipt`)
      .then((response) => {
        const number = response.data?.number || 'sans numéro';
        setSpecimen(response.data);
        setNotice(`Quittance ${number}.`);
      })
      .catch(() => setNotice('Aucun spécimen n’est disponible.'));
  }

  function formatWhen(value) {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat('fr-CH', {
      timeZone: 'Europe/Zurich',
      dateStyle: 'short',
      timeStyle: 'short',
    }).format(date);
  }

  function downloadControl() {
    const csv = cashControlCsv(cases);
    setExportPreview(csv);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'controle-caisse.csv';
    link.click();
    URL.revokeObjectURL(url);
  }

  const position = payload.position;

  return (
    <section aria-label={title} className={styles.wrap}>
      <h3>{title}</h3>
      <p>
        {cashDesk
          ? 'Espèces reçues par les chauffeurs, remises et écarts.'
          : 'Consultation des encaissements. Aucune saisie ni confirmation de remise ici.'}
      </p>
      {notice ? <p role="status">{notice}</p> : null}
      {cashDesk && position ? (
        <div className={styles.summary} aria-label="Espèces détenues">
          <p>{`Espèces encaissées : ${money(position.collected_chf)}`}</p>
          <p>{`Corrections : ${money(position.corrections_chf)}`}</p>
          <p>{`Remboursements : ${money(position.refunds_chf)}`}</p>
          <p>{`Remises confirmées : ${money(position.handed_over_chf)}`}</p>
          <p className={styles.amount}>{`Espèces encore détenues : ${money(position.held_chf)}`}</p>
          {(position.drivers || []).map((driver) => (
            <p key={driver.driver_id}>
              {`${driver.driver_label || driver.driver_id} : ${money(driver.held_chf)} encore détenus`}
            </p>
          ))}
        </div>
      ) : null}
      {cashDesk && !position ? (
        <p>Le détail des espèces détenues n’est pas disponible dans cet environnement.</p>
      ) : null}
      <div className={styles.filters}>
        <label>
          Chauffeur
          <input
            aria-label="Filtre chauffeur"
            value={filters.driver}
            onChange={(event) => setFilter('driver', event.target.value)}
          />
        </label>
        <label>
          Du
          <input
            aria-label="Filtre début"
            type="date"
            value={filters.from}
            onChange={(event) => setFilter('from', event.target.value)}
          />
        </label>
        <label>
          Au
          <input
            aria-label="Filtre fin"
            type="date"
            value={filters.to}
            onChange={(event) => setFilter('to', event.target.value)}
          />
        </label>
        <label>
          Statut
          <select
            aria-label="Filtre statut"
            value={filters.status}
            onChange={(event) => setFilter('status', event.target.value)}
          >
            <option value="">Tous</option>
            <option value="prepared">À encaisser</option>
            <option value="received">Réglé</option>
            <option value="returned">Espèces restituées</option>
            <option value="administrative">Vérification nécessaire</option>
            <option value="uncertain">Réponse incertaine</option>
          </select>
        </label>
        <label>
          Opération
          <select
            aria-label="Filtre opération"
            value={filters.operation}
            onChange={(event) => setFilter('operation', event.target.value)}
          >
            <option value="">Toutes</option>
            <option value="received">Réception</option>
            <option value="returned">Restitution</option>
            <option value="administrative">Conservation</option>
          </select>
        </label>
        <label>
          Remise
          <select
            aria-label="Filtre remise"
            value={filters.handover}
            onChange={(event) => setFilter('handover', event.target.value)}
          >
            <option value="">Toutes</option>
            <option value="aucune">Aucune</option>
            <option value="declared">Annoncée</option>
            <option value="confirmed">Reçue</option>
          </select>
        </label>
        <label>
          Écart
          <select
            aria-label="Filtre écart"
            value={filters.discrepancy}
            onChange={(event) => setFilter('discrepancy', event.target.value)}
          >
            <option value="">Tous</option>
            <option value="oui">Avec écart</option>
            <option value="non">Sans écart</option>
          </select>
        </label>
      </div>
      {cashDesk ? (
        <button type="button" onClick={downloadControl}>
          Exporter le contrôle
        </button>
      ) : null}
      {exportPreview ? (
        <pre aria-label="Export de contrôle">{exportPreview}</pre>
      ) : null}
      {specimen?.lines ? (
        <pre aria-label="Quittance">{specimen.lines.join('\n')}</pre>
      ) : null}
      {cases.length === 0 ? <p>Aucun encaissement enregistré.</p> : null}
      <div className={styles.scroll}>
        <ul className={styles.list}>
          {cases.map((item) => {
            const handovers = Array.isArray(item.handovers) ? item.handovers : [];
            const declared = handovers.filter((line) => line.state === 'declared');
            const legacyAmountKnown = amountKnown(item.amount_due_chf)
              || amountKnown(item.amount_received_chf)
              || amountKnown(item.net_chf);
            const showLegacyConfirm = cashDesk
              && declared.length === 0
              && item.handover_state === 'declared'
              && legacyAmountKnown;
            const confirmable = declared.filter((line) => amountKnown(line.amount_chf));
            const unknownHandover = cashDesk && declared.some((line) => !amountKnown(line.amount_chf));
            const received = item.amount_received_chf ?? item.net_chf;
            const corrections = correctionsOf(item);
            const refunds = refundsOf(item);
            return (
              <li key={item.id}>
                <p>{`Réservation ${item.booking_id} — ${item.company_name || 'Entreprise'}`}</p>
                <p className={styles.amount}>
                  {item.amount_due_chf} {item.currency}
                </p>
                <p>{`Patient : ${item.debtor_label || 'Non communiqué'}`}</p>
                <p>{`Chauffeur : ${item.driver_label || item.driver_id || 'Non communiqué'}`}</p>
                <p>{`Montant contractuel : ${money(item.amount_due_chf)}`}</p>
                <p>{`Montant encaissé : ${money(received)}`}</p>
                <p>{`Montant net reconnu : ${money(item.recognized_net_chf)}`}</p>
                <p>{`Imputation : ${allocationLabel(item)}`}</p>
                <p>Reçu : {item.tendered_chf || '—'} CHF</p>
                <p>Monnaie : {item.change_chf || '—'} CHF</p>
                <p>Net : {item.net_chf || '—'} CHF</p>
                <p>{cashBusinessState(item)}</p>
                <p>Date : {formatWhen(item.created_at)}</p>
                {corrections.length === 0 ? <p>Aucune correction</p> : corrections.map((line) => (
                  <p key={`correction-${line.id}`}>{`Correction : ${money(line.net_chf)}`}</p>
                ))}
                {refunds.map((line) => (
                  <p key={`refund-${line.id}`}>{`Remboursement : ${money(line.net_chf)}`}</p>
                ))}
                {handovers.length === 0 ? (
                  <p>{`Remise : ${handoverLabel(item.handover_state)}`}</p>
                ) : handovers.map((line) => (
                  <p key={line.id}>
                    {`${handoverLabel(line.state)} : ${money(line.amount_chf)}`}
                  </p>
                ))}
                {(item.reconciliations || []).length === 0 ? <p>Aucun écart enregistré</p> : null}
                {(item.reconciliations || []).map((line) => (
                  <p key={line.id}>
                    {line.status === 'regularisation'
                      ? `Régularisation : ${money(line.variance_chf)}`
                      : `Écart à contrôler : ${money(line.variance_chf)}`}
                    {line.open_variance_chf != null && line.status === 'constate'
                      ? ` — reste ${money(line.open_variance_chf)}`
                      : ''}
                    {line.reason ? ` — ${line.reason}` : ''}
                  </p>
                ))}
                {item.state === 'administrative' && item.tendered_chf ? (
                  <p>{`Espèces détenues ${item.tendered_chf} CHF. Vérification nécessaire.`}</p>
                ) : null}
                {showLegacyConfirm ? (
                  <button type="button" onClick={() => confirm(item.id)} disabled={!canManageCash}>
                    Confirmer les fonds reçus
                  </button>
                ) : null}
                {cashDesk && confirmable.map((line) => (
                  <button
                    key={`confirm-${line.id}`}
                    type="button"
                    onClick={() => confirm(item.id, line)}
                    disabled={!canManageCash}
                  >
                    {`Confirmer la remise de ${line.amount_chf} CHF`}
                  </button>
                ))}
                {unknownHandover ? (
                  <p>Confirmation impossible : le montant de la remise n’est pas communiqué.</p>
                ) : null}
                {cashDesk && !canManageCash && (showLegacyConfirm || confirmable.length > 0) ? (
                  <p>La confirmation est réservée au responsable de l’entreprise.</p>
                ) : null}
                {cashDesk && canManageCash && handovers
                  .filter((line) => line.state === 'confirmed' && !hasConstat(item, line.id))
                  .map((line) => (
                    <div key={`count-${line.id}`} className={styles.actions}>
                      <label>
                        Fonds comptés
                        <input
                          aria-label={`Fonds comptés ${line.id}`}
                          value={counted[line.id] || ''}
                          onChange={(event) => setCounted((current) => ({
                            ...current,
                            [line.id]: event.target.value,
                          }))}
                        />
                      </label>
                      <label>
                        Motif
                        <input
                          aria-label={`Motif ${line.id}`}
                          value={motifs[line.id] || ''}
                          onChange={(event) => setMotifs((current) => ({
                            ...current,
                            [line.id]: event.target.value,
                          }))}
                        />
                      </label>
                      <button type="button" onClick={() => reconcile(item.id, line.id)}>
                        Enregistrer l’écart
                      </button>
                    </div>
                  ))}
                {cashDesk && canManageCash && handovers.length === 0
                  && item.handover_state === 'confirmed'
                  && item.reconciliation_variance_chf == null ? (
                    <div className={styles.actions}>
                      <label>
                        Fonds comptés
                        <input
                          aria-label={`Fonds comptés ${item.id}`}
                          value={counted[item.id] || ''}
                          onChange={(event) => setCounted((current) => ({
                            ...current,
                            [item.id]: event.target.value,
                          }))}
                        />
                      </label>
                      <button type="button" onClick={() => reconcile(item.id)}>
                        Enregistrer l’écart
                      </button>
                    </div>
                  ) : null}
                {item.receipt_number ? (
                  <p>{`Quittance ${item.receipt_number}`}</p>
                ) : null}
                {item.balance_chf && item.balance_chf !== '0.00' ? (
                  <p>{`Partiellement réglé. Solde restant ${item.balance_chf} CHF.`}</p>
                ) : null}
                {item.receipt_number ? (
                  <button type="button" onClick={() => receipt(item.id)}>
                    Voir la quittance
                  </button>
                ) : null}
                <button type="button" onClick={() => setOpenId(openId === item.id ? null : item.id)}>
                  Historique
                </button>
                {openId === item.id ? (
                  <ol>
                    {(item.history || []).map((event, index) => (
                      <li key={`${item.id}-${index}`}>
                        {`${cashHistoryLabel(event.action)} — ${formatWhen(event.at)}`}
                      </li>
                    ))}
                  </ol>
                ) : null}
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
