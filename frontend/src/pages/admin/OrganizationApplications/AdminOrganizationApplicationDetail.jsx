import React, { useCallback, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  approveOrganizationApplication,
  decideOrganizationApplication,
  fetchOrganizationApplication,
  searchExistingOrganizations,
} from '../../../services/adminOrganizationApplicationService';
import AdminActionDialog from '../components/AdminActionDialog';
import { adminPaths } from '../routing/adminRoutePaths';
import shell from '../adminShell.module.css';
import styles from './AdminOrganizationApplications.module.css';

const DECISIONS = {
  start_review: {
    title: 'Prendre en charge',
    description: 'Le dossier passe en examen. Aucun portail n’est ouvert.',
    confirmationLabel: 'Prendre en charge',
  },
  request_info: {
    title: 'Demander un complément',
    description: 'Le demandeur est informé. Le délai pourra plus tard tenir compte de cette suspension.',
    confirmationLabel: 'Envoyer la demande',
    reason: { required: true, label: 'Complément attendu', minLength: 10 },
  },
  record_info: {
    title: 'Enregistrer le complément reçu',
    description: 'Cela ne reprend pas encore l’examen.',
    confirmationLabel: 'Enregistrer',
    reason: { required: true, label: 'Ce qui a été reçu', minLength: 10 },
  },
  resume_review: {
    title: 'Reprendre l’examen',
    description: 'Le dossier revient en cours d’examen, sans ouvrir de portail.',
    confirmationLabel: 'Reprendre',
  },
  refuse: {
    title: 'Refuser la demande',
    description: 'Le refus est motivé et tracé. Aucune organisation n’est créée.',
    confirmationLabel: 'Refuser la demande',
    confirmText: 'REFUSER',
    danger: true,
    reason: { required: true, label: 'Motif du refus', minLength: 10 },
  },
  approve: {
    title: 'Approuver et activer',
    description: 'Cette action prépare l’organisation et l’habilitation. Elle reste fermée tant que le verrou n’est pas ouvert.',
    confirmationLabel: 'Approuver et activer',
    confirmText: 'APPROUVER',
  },
};

function formatWhen(value) {
  if (!value) return '—';
  return new Date(value).toLocaleString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function AdminOrganizationApplicationDetail() {
  const { public_id: adminId, applicationId } = useParams();
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [decision, setDecision] = useState(null);
  const [note, setNote] = useState('');
  const [identityConfirmed, setIdentityConfirmed] = useState('yes');
  const [organizationConfirmed, setOrganizationConfirmed] = useState('yes');
  const [search, setSearch] = useState('');
  const [matches, setMatches] = useState({ companies: [], institutions: [] });

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setDetail(await fetchOrganizationApplication(applicationId));
    } catch {
      setDetail(null);
      setError('Impossible de charger ce dossier.');
    } finally {
      setLoading(false);
    }
  }, [applicationId]);

  useEffect(() => {
    load();
  }, [load]);

  const runDecision = async ({ reason } = {}) => {
    const updated = decision === 'approve'
      ? await approveOrganizationApplication(applicationId)
      : await decideOrganizationApplication(applicationId, {
        action: decision,
        note: reason || undefined,
      });
    setDetail(updated);
    setDecision(null);
  };

  const recordCheck = async (action, confirmed) => {
    setError('');
    try {
      setDetail(await decideOrganizationApplication(applicationId, {
        action,
        confirmed,
        note,
      }));
      setNote('');
    } catch (requestError) {
      setError(requestError.message);
    }
  };

  const lookUp = async (event) => {
    event.preventDefault();
    setError('');
    try {
      setMatches(await searchExistingOrganizations(applicationId, search));
    } catch {
      setError('La recherche d’organisation a échoué.');
    }
  };

  const noteMatch = async (payload) => {
    setError('');
    try {
      setDetail(await decideOrganizationApplication(applicationId, {
        action: 'note_existing_organization',
        note,
        ...payload,
      }));
    } catch (requestError) {
      setError(requestError.message);
    }
  };

  const dialog = decision ? DECISIONS[decision] : null;
  const available = [];
  if (detail?.status === 'received') available.push('start_review');
  if (detail?.status === 'in_review') available.push('request_info');
  if (detail?.status === 'needs_info') available.push('record_info', 'resume_review');
  if (detail?.actionable) available.push('refuse');

  return (
    <main className={shell.content}>
      <p>
        <Link className={styles.linkButton} to={adminPaths.partnersOrganizationApplications(adminId)}>
          Retour à la file
        </Link>
      </p>
      {loading ? <p className={styles.loading}>Chargement du dossier…</p> : null}
      {error ? <p className={styles.alert} role="alert">{error}</p> : null}
      {detail ? (
        <>
          <header className={styles.pageHeader}>
            <div>
              <p className={styles.meta}>{detail.kind_label}</p>
              <h1>{detail.organization_name}</h1>
              <p className={styles.lead}>
                {detail.incomplete
                  ? 'Dossier incomplet : l’adresse e-mail n’est pas confirmée.'
                  : detail.status_label}
                {' · '}
                {detail.queue_label}
              </p>
            </div>
            <button type="button" className={styles.ghost} onClick={load}>Actualiser</button>
          </header>

          <section className={styles.section} aria-labelledby="dossier-facts">
            <h2 id="dossier-facts">Dossier</h2>
            <dl className={styles.facts}>
              <div>
                <dt>Représentant</dt>
                <dd>{detail.applicant_first_name} {detail.applicant_last_name}</dd>
              </div>
              <div>
                <dt>Fonction</dt>
                <dd>{detail.applicant_function}</dd>
              </div>
              <div>
                <dt>E-mail</dt>
                <dd className={styles.email}>{detail.applicant_email}</dd>
              </div>
              <div>
                <dt>Téléphone</dt>
                <dd>{detail.applicant_phone}</dd>
              </div>
              <div>
                <dt>Adresse</dt>
                <dd>{detail.street}, {detail.postal_code} {detail.city}</dd>
              </div>
              {detail.ide_number ? (
                <div>
                  <dt>IDE</dt>
                  <dd>{detail.ide_number}</dd>
                </div>
              ) : null}
              {detail.establishment_category ? (
                <div>
                  <dt>Catégorie</dt>
                  <dd>{detail.establishment_category}</dd>
                </div>
              ) : null}
            </dl>
            {detail.applicant_review_note ? <p className={styles.hint}>{detail.applicant_review_note}</p> : null}
          </section>

          <section className={styles.section} aria-labelledby="dossier-sla">
            <h2 id="dossier-sla">Délai</h2>
            <p className={styles.lead}>{detail.sla.notice}</p>
            <p className={styles.hint}>Le calendrier des heures administratives n’est pas encore configuré.</p>
            {detail.sla.suspensions.length > 0 ? (
              <ul className={styles.history}>
                {detail.sla.suspensions.map((item) => (
                  <li key={`${item.from}-${item.until || 'ouverte'}`}>
                    Suspension depuis {formatWhen(item.from)}
                    {item.until ? ` jusqu’au ${formatWhen(item.until)}` : ', encore ouverte'}
                  </li>
                ))}
              </ul>
            ) : null}
          </section>

          <section className={styles.section} aria-labelledby="dossier-documents">
            <h2 id="dossier-documents">Justificatifs</h2>
            <p className={styles.hint}>{detail.documents_note}</p>
          </section>

          <section className={styles.section} aria-labelledby="dossier-checks">
            <h2 id="dossier-checks">Vérifications</h2>
            <p className={styles.hint}>
              Identité : {detail.checks.identity.confirmed === null ? 'non consignée' : detail.checks.identity.confirmed ? 'confirmée' : 'non confirmée'}.
              Organisation : {detail.checks.organization.confirmed === null ? 'non consignée' : detail.checks.organization.confirmed ? 'confirmée' : 'non confirmée'}.
            </p>
            <label className={styles.field}>
              <span>Observation</span>
              <textarea value={note} onChange={(event) => setNote(event.target.value)} />
            </label>
            <div className={styles.checkRow}>
              <label>
                <input
                  type="radio"
                  name="identity"
                  checked={identityConfirmed === 'yes'}
                  onChange={() => setIdentityConfirmed('yes')}
                />
                Identité confirmée
              </label>
              <label>
                <input
                  type="radio"
                  name="identity"
                  checked={identityConfirmed === 'no'}
                  onChange={() => setIdentityConfirmed('no')}
                />
                Identité non confirmée
              </label>
              <button type="button" className={styles.ghost} onClick={() => recordCheck('check_identity', identityConfirmed === 'yes')}>
                Consigner l’identité
              </button>
            </div>
            <div className={styles.checkRow}>
              <label>
                <input
                  type="radio"
                  name="organization"
                  checked={organizationConfirmed === 'yes'}
                  onChange={() => setOrganizationConfirmed('yes')}
                />
                Organisation confirmée
              </label>
              <label>
                <input
                  type="radio"
                  name="organization"
                  checked={organizationConfirmed === 'no'}
                  onChange={() => setOrganizationConfirmed('no')}
                />
                Organisation non confirmée
              </label>
              <button type="button" className={styles.ghost} onClick={() => recordCheck('check_organization', organizationConfirmed === 'yes')}>
                Consigner l’organisation
              </button>
            </div>
            <button
              type="button"
              className={styles.ghost}
              onClick={() => recordCheck('observe', true)}
            >
              Ajouter l’observation
            </button>
          </section>

          <section className={styles.section} aria-labelledby="dossier-match">
            <h2 id="dossier-match">Organisation déjà connue</h2>
            <p className={styles.hint}>
              La correspondance est notée. Elle ne crée pas de société, d’institution ni d’habilitation.
            </p>
            <form className={styles.searchRow} onSubmit={lookUp}>
              <label className={styles.field}>
                <span>Recherche</span>
                <input value={search} onChange={(event) => setSearch(event.target.value)} />
              </label>
              <button type="submit" className={styles.button}>Rechercher</button>
            </form>
            <ul className={styles.history}>
              {matches.companies.map((company) => (
                <li key={`company-${company.id}`}>
                  <button type="button" className={styles.ghost} onClick={() => noteMatch({ company_id: company.id })}>
                    Noter l’entreprise {company.name}
                  </button>
                </li>
              ))}
              {matches.institutions.map((institution) => (
                <li key={`institution-${institution.id}`}>
                  <button type="button" className={styles.ghost} onClick={() => noteMatch({ institution_id: institution.id })}>
                    Noter l’institution {institution.name}
                  </button>
                </li>
              ))}
            </ul>
          </section>

          <section className={styles.section} aria-labelledby="dossier-actions">
            <h2 id="dossier-actions">Instruction</h2>
            {detail.incomplete ? (
              <p className={styles.hint}>Aucune décision n’est possible tant que l’adresse n’est pas confirmée.</p>
            ) : null}
            <div className={styles.actions}>
              {available.map((action) => (
                <button
                  key={action}
                  type="button"
                  className={action === 'refuse' ? `${styles.button} ${styles.danger}` : styles.button}
                  onClick={() => setDecision(action)}
                >
                  {DECISIONS[action].title}
                </button>
              ))}
            </div>
          </section>

          <section className={`${styles.section} ${styles.locked}`} aria-labelledby="dossier-approval">
            <h2 id="dossier-approval">Approbation future</h2>
            <p className={styles.lead}>{detail.approval.reason}</p>
            <button
              type="button"
              className={styles.button}
              disabled={!detail.approval?.enabled}
              onClick={() => {
                if (detail.approval?.enabled) setDecision('approve');
              }}
            >
              Approuver et activer
            </button>
          </section>

          <section className={styles.section} aria-labelledby="dossier-history">
            <h2 id="dossier-history">Historique</h2>
            <ol className={styles.history}>
              {detail.events.map((event) => (
                <li key={`${event.type}-${event.at}-${event.note || ''}`}>
                  <strong>{event.label}</strong>
                  {event.actor ? ` · ${event.actor}` : ''}
                  {event.visibility === 'internal' ? ' · interne' : ''}
                  <div>{formatWhen(event.at)}</div>
                  {event.note ? <div className={styles.email}>{event.note}</div> : null}
                </li>
              ))}
            </ol>
          </section>
        </>
      ) : null}

      <AdminActionDialog
        open={Boolean(dialog)}
        title={dialog?.title || ''}
        description={dialog?.description || ''}
        confirmationLabel={dialog?.confirmationLabel || 'Confirmer'}
        confirmText={dialog?.confirmText}
        danger={dialog?.danger}
        reason={dialog?.reason}
        onClose={() => setDecision(null)}
        onConfirm={runDecision}
      />
    </main>
  );
}
