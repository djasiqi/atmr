import React, { useCallback, useEffect, useState } from 'react';
import apiClient from '../../../../utils/apiClient';
import styles from './ClientReadView.module.css';

const STATUS_LABELS = {
  draft: 'Brouillon',
  awaiting_signature: 'En attente de signature',
  signed_pending_validation: 'Signée — validation requise',
  applicable: 'Applicable (effet juridique fermé)',
  expired: 'Expirée',
  revoked: 'Révoquée',
  replaced: 'Remplacée',
};

/**
 * Convention de transport Client réelle (hors démo).
 * Différé réel fermé tant que l'activation juridique n'est pas ouverte.
 */
export default function ClientConventionPanel({ clientId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState(null);
  const [debtorLabel, setDebtorLabel] = useState('');

  const load = useCallback(() => {
    if (!clientId) return;
    setLoading(true);
    apiClient
      .get(`/companies/me/clients/${clientId}/conventions`)
      .then((response) => setData(response.data))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [clientId]);

  useEffect(() => {
    load();
  }, [load]);

  const createDraft = async () => {
    try {
      await apiClient.post(`/companies/me/clients/${clientId}/conventions`, {
        debtor_label: debtorLabel.trim() || `Client ${clientId}`,
        tariff_version: 'standard',
        service_scope: 'Transports de cette entreprise.',
        document_version: 'v1',
        idempotency_key: `ui-create-${clientId}-${Date.now()}`,
      });
      setNotice('Brouillon de convention créé.');
      load();
    } catch (error) {
      setNotice(error?.response?.data?.error || 'Création refusée.');
    }
  };

  const act = async (conventionId, action, body = {}) => {
    try {
      await apiClient.post(
        `/companies/me/clients/${clientId}/conventions/${conventionId}/${action}`,
        {
          idempotency_key: `ui-${action}-${conventionId}-${Date.now()}`,
          ...body,
        }
      );
      setNotice(`Action « ${action} » effectuée.`);
      load();
    } catch (error) {
      setNotice(error?.response?.data?.error || 'Action refusée.');
    }
  };

  if (loading) {
    return <p className={styles.section}>Chargement des conventions…</p>;
  }

  const active = data?.active;
  const versions = data?.conventions || [];

  return (
    <section className={styles.section} aria-label="Conventions de transport">
      <h3 className={styles.sectionTitle}>Convention de transport</h3>
      <p>
        Document entreprise ↔ client pour facturation différée. Distinct des CGV,
        des accords plateforme et des prises en charge institutionnelles.
        Différé réel fermé jusqu&apos;à validation juridique.
      </p>
      {notice ? <p role="status">{notice}</p> : null}

      {!active ? (
        <div>
          <label htmlFor={`debtor-${clientId}`}>Débiteur (libellé)</label>
          <input
            id={`debtor-${clientId}`}
            value={debtorLabel}
            onChange={(event) => setDebtorLabel(event.target.value)}
            placeholder="Nom du débiteur contractuel"
          />
          <button type="button" onClick={createDraft}>
            Créer un brouillon
          </button>
        </div>
      ) : (
        <div>
          <p>
            <strong>Statut :</strong> {STATUS_LABELS[active.status] || active.status}
          </p>
          <p>
            <strong>Version :</strong> {active.version_number} / {active.document_version}
          </p>
          <p>
            <strong>Débiteur :</strong> {active.debtor_label}
            {active.invoice_recipient_label
              ? ` · Destinataire facture : ${active.invoice_recipient_label}`
              : ''}
          </p>
          <p>
            <strong>Empreinte :</strong> {active.document_sha256 || '—'}
          </p>
          <p>
            <strong>Différé autorisé :</strong>{' '}
            {active.authorizes_deferred_billing ? 'oui' : 'non'} ({active.deferred_reason})
          </p>
          <div>
            {active.status === 'draft' ? (
              <button type="button" onClick={() => act(active.id, 'submit')}>
                Soumettre à signature
              </button>
            ) : null}
            {active.status === 'awaiting_signature' || active.status === 'draft' ? (
              <button
                type="button"
                onClick={() =>
                  act(active.id, 'paper', {
                    paper_scan_reference: `scan-admin-${active.id}`,
                    signer_name: active.debtor_label,
                    signer_kind: 'client',
                  })
                }
              >
                Enregistrer scan papier
              </button>
            ) : null}
            {active.status === 'signed_pending_validation' ? (
              <button type="button" onClick={() => act(active.id, 'validate')}>
                Marquer applicable (sans effet juridique)
              </button>
            ) : null}
            {active.archive_reference ? (
              <a
                href={`/api/v1/companies/me/clients/${clientId}/conventions/${active.id}/document.pdf`}
                target="_blank"
                rel="noreferrer"
              >
                Télécharger PDF figé
              </a>
            ) : null}
            {['draft', 'awaiting_signature', 'signed_pending_validation', 'applicable'].includes(
              active.status
            ) ? (
              <button
                type="button"
                onClick={() =>
                  act(active.id, 'revoke', { reason: 'Révocation administrative UI' })
                }
              >
                Révoquer
              </button>
            ) : null}
          </div>
        </div>
      )}

      {versions.length > 1 ? (
        <details>
          <summary>Historique des versions ({versions.length})</summary>
          <ul>
            {versions.map((row) => (
              <li key={row.id}>
                v{row.version_number} — {STATUS_LABELS[row.status] || row.status}
                {row.active ? ' (active)' : ''}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
