import React, { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import {
  fetchAdminDemoRequests,
  provisionDemoAccess,
  resendDemoAccess,
  revokeDemoAccess,
  updateDemoRequestStatus,
} from '../../../services/adminDemoService';
import styles from './AdminDemoRequests.module.css';
import shell from '../adminShell.module.css';

const REQUEST_STATUS_META = {
  new: { label: 'Nouvelle', tone: 'neutral' },
  contacted: { label: 'Contactee', tone: 'info' },
  qualified: { label: 'Validee', tone: 'success' },
  rejected: { label: 'Refusee', tone: 'danger' },
};

const ACCESS_STATUS_META = {
  pending: { label: 'Aucun acces', tone: 'neutral' },
  active: { label: 'Actif', tone: 'success' },
  expired: { label: 'Expire', tone: 'warning' },
  revoked: { label: 'Revoque', tone: 'danger' },
};

const getRequestStatusMeta = (status) => REQUEST_STATUS_META[status] || { label: status || '-', tone: 'neutral' };
const getAccessStatusMeta = (status) => ACCESS_STATUS_META[status] || { label: status || '-', tone: 'neutral' };

const formatDateTime = (value) => {
  if (!value) return '-';
  return new Date(value).toLocaleString('fr-FR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const ORG_TYPE_OPTIONS = [
  { value: 'institution', label: 'Institution' },
  { value: 'ems', label: 'EMS' },
  { value: 'clinic', label: 'Clinique' },
  { value: 'hospital', label: 'Hopital' },
  { value: 'curatorship', label: 'Curatelle' },
  { value: 'transport_company', label: 'Entreprise de transport' },
  { value: 'other', label: 'Autre' },
];

const TEMPLATE_OPTIONS = [
  { value: 'institution_demo', label: 'Institution demo' },
  { value: 'ems_demo', label: 'EMS demo' },
  { value: 'clinic_demo', label: 'Clinique demo' },
  { value: 'transport_company_demo', label: 'Transporteur demo' },
  { value: 'generic_demo', label: 'Generic demo' },
];

const PERSONA_OPTIONS = [
  { value: 'institution', label: 'Persona institution' },
  { value: 'transport_company', label: 'Persona transporteur' },
  { value: 'generic', label: 'Persona generic' },
];

const GUIDE_OPTIONS = [
  { value: 'institution_quickstart', label: 'Guide institution quickstart' },
  { value: 'transport_dispatch_quickstart', label: 'Guide transport dispatch quickstart' },
  { value: 'generic_quickstart', label: 'Guide generic quickstart' },
];

const buildDemoLoginEmail = (email) => {
  const raw = String(email || '').trim().toLowerCase();
  if (!raw) return 'demo-user@demo.local';
  if (!raw.includes('@')) return `demo-${raw}@demo.local`;
  const [local, domain] = raw.split('@');
  const safeLocal = local.startsWith('demo-') ? local : `demo-${local || 'user'}`;
  return `${safeLocal}@${domain || 'demo.local'}`;
};

const splitName = (fullName) => {
  const cleaned = String(fullName || '').trim().replace(/\s+/g, ' ');
  if (!cleaned) return { firstName: '', lastName: '' };
  const parts = cleaned.split(' ');
  if (parts.length === 1) return { firstName: parts[0], lastName: '' };
  return { firstName: parts[0], lastName: parts.slice(1).join(' ') };
};

const isInstitutionType = (orgType) =>
  ['institution', 'ems', 'clinic', 'hospital', 'curatorship'].includes(
    String(orgType || '').toLowerCase()
  );

const defaultProvisionFromRequest = (item) => {
  const { firstName, lastName } = splitName(item?.name);
  const organizationType = String(item?.organization_type || '').toLowerCase() || 'institution';
  const institutionJourney = isInstitutionType(organizationType);
  return {
    organization_name: item?.organization || '',
    organization_type: organizationType,
    organization_address: '',
    organization_contact_phone: item?.phone || '',
    organization_contact_email: item?.email || '',
    workspace_display_name: item?.organization || '',
    demo_login_email: buildDemoLoginEmail(item?.email),
    user_first_name: firstName,
    user_last_name: lastName,
    user_phone: item?.phone || '',
    user_role: institutionJourney ? 'institution_admin' : 'company_admin',
    provision_template: institutionJourney ? 'institution_demo' : 'transport_company_demo',
    demo_persona: institutionJourney ? 'institution' : 'transport_company',
    guide_variant: institutionJourney
      ? 'institution_quickstart'
      : 'transport_dispatch_quickstart',
    seed_context: {
      volume_range: item?.volume_range || '',
      timing: item?.timing || '',
      preferred_slot: item?.preferred_slot || '',
      preferred_period: item?.preferred_period || '',
      integration_required: item?.integration_required || '',
    },
    internal_admin_notes: '',
    visible_demo_notes: item?.comment || '',
    workspace_seed_notes: `timing=${item?.timing || '-'}; volume=${item?.volume_range || '-'}; slot=${
      item?.preferred_slot || '-'
    }; period=${item?.preferred_period || '-'}`,
  };
};

const AdminDemoRequests = () => {
  const [searchParams] = useSearchParams();
  const statusFilter = searchParams.get('status') || 'all';

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [items, setItems] = useState([]);
  const [busyKey, setBusyKey] = useState('');
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [provisionForm, setProvisionForm] = useState(null);

  const displayItems = useMemo(() => {
    if (statusFilter === 'new') {
      return items.filter((i) => i.status === 'new');
    }
    return items;
  }, [items, statusFilter]);

  const load = async (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
      setError('');
    }
    try {
      const data = await fetchAdminDemoRequests();
      setItems(data);
    } catch (err) {
      if (showLoading) {
        setError("Impossible de charger les demandes de demonstration.");
      }
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    load();
  }, []);

  // Rafraîchissement automatique toutes les 60s pour afficher les nouvelles demandes
  useEffect(() => {
    const interval = setInterval(() => load(false), 60_000);
    return () => clearInterval(interval);
  }, []);

  const runAction = async (key, action) => {
    setBusyKey(key);
    try {
      await action();
      await load();
    } catch (err) {
      setError(err?.response?.data?.message || "Action demo indisponible.");
    } finally {
      setBusyKey('');
    }
  };

  const metrics = items.reduce(
    (acc, item) => {
      const accessStatus = item.latest_access?.status || 'pending';
      acc.total += 1;
      if (item.status === 'new') acc.newCount += 1;
      if (accessStatus === 'active') acc.activeCount += 1;
      if (accessStatus === 'expired') acc.expiredCount += 1;
      if (accessStatus === 'revoked') acc.revokedCount += 1;
      return acc;
    },
    { total: 0, newCount: 0, activeCount: 0, expiredCount: 0, revokedCount: 0 }
  );

  const qualityChecks = useMemo(() => {
    if (!provisionForm) return [];
    return [
      {
        label: 'Nom organisation rempli',
        ok: Boolean(provisionForm.organization_name?.trim()),
      },
      {
        label: 'Type organisation coherent',
        ok: Boolean(provisionForm.organization_type?.trim()),
      },
      {
        label: 'Login demo isole (demo-...)',
        ok: /^demo-.+@.+$/.test(String(provisionForm.demo_login_email || '').trim().toLowerCase()),
      },
      {
        label: 'Contact principal nom/prenom',
        ok: Boolean(
          provisionForm.user_first_name?.trim() && provisionForm.user_last_name?.trim()
        ),
      },
      {
        label: 'Template + persona + guide choisis',
        ok: Boolean(
          provisionForm.provision_template &&
            provisionForm.demo_persona &&
            provisionForm.guide_variant
        ),
      },
      {
        label: 'Adresse institution recommandee',
        ok: !isInstitutionType(provisionForm.organization_type) ||
          Boolean(provisionForm.organization_address?.trim()),
        warningOnly: true,
      },
    ];
  }, [provisionForm]);

  const openProvisionAssistant = (item) => {
    setSelectedRequest(item);
    setProvisionForm(defaultProvisionFromRequest(item));
    setError('');
  };

  const closeProvisionAssistant = () => {
    setSelectedRequest(null);
    setProvisionForm(null);
  };

  const updateProvisionField = (event) => {
    const { name, value } = event.target;
    setProvisionForm((prev) => (prev ? { ...prev, [name]: value } : prev));
  };

  const updateSeedContextField = (event) => {
    const { name, value } = event.target;
    setProvisionForm((prev) =>
      prev
        ? {
            ...prev,
            seed_context: {
              ...(prev.seed_context || {}),
              [name]: value,
            },
          }
        : prev
    );
  };

  const handleOrganizationAddressSelect = (item) => {
    const nextAddress = item?.label || item?.address || '';
    setProvisionForm((prev) =>
      prev
        ? {
            ...prev,
            organization_address: nextAddress,
          }
        : prev
    );
  };

  const submitProvisionAssistant = async () => {
    if (!selectedRequest || !provisionForm) return;
    if (!provisionForm.organization_name?.trim()) {
      setError("Le nom de l'organisation est requis.");
      return;
    }
    if (!provisionForm.demo_login_email?.trim()) {
      setError('Le login demo est requis.');
      return;
    }
    await runAction(`p-${selectedRequest.id}`, async () => {
      await provisionDemoAccess(selectedRequest.id, provisionForm);
      closeProvisionAssistant();
    });
  };

  return (
    <>
      <main className={shell.content}>
          <header className={styles.pageHeader}>
            <div>
              <h1>Demandes de demonstration</h1>
              <p className={styles.subtext}>
                Gestion des acces demo 24h : provision, renvoi et revocation.
              </p>
            </div>
            <button
              type="button"
              onClick={() => load()}
              disabled={loading}
              className={styles.refreshButton}
              title="Rafraîchir la liste"
            >
              {loading ? 'Chargement...' : 'Rafraîchir'}
            </button>
          </header>

          {!loading && !error && (
            <section className={styles.metricsGrid} aria-label="Synthese demandes demo">
              <article className={styles.metricCard}>
                <span>Total</span>
                <strong>{metrics.total}</strong>
              </article>
              <article className={styles.metricCard}>
                <span>Nouvelles demandes</span>
                <strong>{metrics.newCount}</strong>
              </article>
              <article className={styles.metricCard}>
                <span>Acces actifs</span>
                <strong>{metrics.activeCount}</strong>
              </article>
              <article className={styles.metricCard}>
                <span>Acces expires</span>
                <strong>{metrics.expiredCount}</strong>
              </article>
              <article className={styles.metricCard}>
                <span>Acces revoques</span>
                <strong>{metrics.revokedCount}</strong>
              </article>
            </section>
          )}

          {loading && <p className={styles.info}>Chargement des demandes...</p>}
          {error && <p className={styles.error}>{error}</p>}

          {!loading && (
            <div className={styles.tableWrap}>
              {items.length === 0 ? (
                <div className={styles.emptyState}>
                  <h2>Aucune demande</h2>
                  <p>Aucune demande de demonstration n est disponible pour le moment.</p>
                </div>
              ) : displayItems.length === 0 ? (
                <div className={styles.emptyState}>
                  <h2>Aucune demande dans ce filtre</h2>
                  <p>Aucune demande ne correspond au filtre actuel (statut).</p>
                </div>
              ) : (
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Demandeur</th>
                      <th>Organisation</th>
                      <th>Statut demande</th>
                      <th>Statut acces</th>
                      <th>Expiration</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayItems.map((item) => {
                      const access = item.latest_access;
                      const accessStatus = access?.status || 'pending';
                      const hasActiveAccess = accessStatus === 'active';
                      const requestMeta = getRequestStatusMeta(item.status);
                      const accessMeta = getAccessStatusMeta(accessStatus);
                      return (
                        <tr key={item.id}>
                          <td className={styles.idCell}>#{item.id}</td>
                          <td>
                            <div className={styles.requesterName}>{item.name || '-'}</div>
                            <small className={styles.requesterEmail}>{item.email || '-'}</small>
                          </td>
                          <td>
                            <div>{item.organization || '-'}</div>
                            <small className={styles.requesterEmail}>{item.organization_type || '-'}</small>
                          </td>
                          <td>
                            <span className={`${styles.badge} ${styles[`tone${requestMeta.tone}`]}`}>
                              {requestMeta.label}
                            </span>
                          </td>
                          <td>
                            <span className={`${styles.badge} ${styles[`tone${accessMeta.tone}`]}`}>
                              {accessMeta.label}
                            </span>
                          </td>
                          <td>{formatDateTime(access?.demo_expires_at)}</td>
                          <td className={styles.actions}>
                            <button
                              type="button"
                              className={styles.primaryButton}
                              disabled={busyKey === `p-${item.id}` || hasActiveAccess}
                              onClick={() => openProvisionAssistant(item)}
                            >
                              Approuver et envoyer acces
                            </button>
                            <button
                              type="button"
                              className={styles.dangerButton}
                              disabled={busyKey === `x-${item.id}` || hasActiveAccess}
                              onClick={() =>
                                runAction(`x-${item.id}`, () => updateDemoRequestStatus(item.id, 'rejected'))
                              }
                            >
                              Refuser
                            </button>
                            <button
                              type="button"
                              className={styles.ghostButton}
                              disabled={busyKey === `r-${access?.id}` || !access?.id || !hasActiveAccess}
                              onClick={() =>
                                runAction(`r-${access?.id}`, () => resendDemoAccess(access.id))
                              }
                            >
                              Renvoyer
                            </button>
                            <button
                              type="button"
                              className={styles.dangerButton}
                              disabled={busyKey === `v-${access?.id}` || !access?.id || !hasActiveAccess}
                              onClick={() =>
                                runAction(`v-${access?.id}`, () => revokeDemoAccess(access.id))
                              }
                            >
                              Revoquer
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </main>

      {selectedRequest && provisionForm && (
        <div className={styles.modalOverlay} role="dialog" aria-modal="true">
          <div className={styles.modalCard}>
            <header className={styles.modalHeader}>
              <div>
                <h2>Assistant de preparation demo</h2>
                <p>
                  Demande #{selectedRequest.id} - {selectedRequest.name || '-'} ({selectedRequest.email || '-'})
                </p>
              </div>
              <button type="button" className={styles.closeButton} onClick={closeProvisionAssistant}>
                Fermer
              </button>
            </header>

            <section className={styles.modalSection}>
              <h3>Organisation</h3>
              <div className={styles.formGrid}>
                <label>
                  Nom organisation
                  <input
                    name="organization_name"
                    value={provisionForm.organization_name}
                    onChange={updateProvisionField}
                  />
                </label>
                <label>
                  Type organisation
                  <select
                    name="organization_type"
                    value={provisionForm.organization_type}
                    onChange={updateProvisionField}
                  >
                    {ORG_TYPE_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className={styles.fullWidth}>
                  Adresse
                  <AddressAutocomplete
                    inputId="admin_demo_organization_address"
                    name="organization_address"
                    value={provisionForm.organization_address || ''}
                    onChange={updateProvisionField}
                    onSelect={handleOrganizationAddressSelect}
                    placeholder="Adresse officielle de l'institution ou de l'entreprise"
                  />
                </label>
                <label>
                  Email contact (visible)
                  <input
                    type="email"
                    name="organization_contact_email"
                    value={provisionForm.organization_contact_email}
                    onChange={updateProvisionField}
                  />
                </label>
                <label>
                  Telephone contact
                  <input
                    name="organization_contact_phone"
                    value={provisionForm.organization_contact_phone}
                    onChange={updateProvisionField}
                  />
                </label>
              </div>
            </section>

            <section className={styles.modalSection}>
              <h3>Contact principal et login demo</h3>
              <div className={styles.formGrid}>
                <label>
                  Prenom
                  <input
                    name="user_first_name"
                    value={provisionForm.user_first_name}
                    onChange={updateProvisionField}
                  />
                </label>
                <label>
                  Nom
                  <input
                    name="user_last_name"
                    value={provisionForm.user_last_name}
                    onChange={updateProvisionField}
                  />
                </label>
                <label>
                  Telephone utilisateur
                  <input name="user_phone" value={provisionForm.user_phone} onChange={updateProvisionField} />
                </label>
                <label>
                  Email login demo (technique)
                  <input
                    name="demo_login_email"
                    value={provisionForm.demo_login_email}
                    onChange={updateProvisionField}
                  />
                </label>
              </div>
            </section>

            <section className={styles.modalSection}>
              <h3>Type de demo et guide</h3>
              <div className={styles.formGrid}>
                <label>
                  Template
                  <select
                    name="provision_template"
                    value={provisionForm.provision_template}
                    onChange={updateProvisionField}
                  >
                    {TEMPLATE_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Persona
                  <select
                    name="demo_persona"
                    value={provisionForm.demo_persona}
                    onChange={updateProvisionField}
                  >
                    {PERSONA_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className={styles.fullWidth}>
                  Guide variant
                  <select
                    name="guide_variant"
                    value={provisionForm.guide_variant}
                    onChange={updateProvisionField}
                  >
                    {GUIDE_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </section>

            <section className={styles.modalSection}>
              <h3>Parametres seed</h3>
              <div className={styles.formGrid}>
                <label>
                  Volume
                  <input
                    name="volume_range"
                    value={provisionForm.seed_context?.volume_range || ''}
                    onChange={updateSeedContextField}
                  />
                </label>
                <label>
                  Timing
                  <input
                    name="timing"
                    value={provisionForm.seed_context?.timing || ''}
                    onChange={updateSeedContextField}
                  />
                </label>
                <label>
                  Creneau
                  <input
                    name="preferred_slot"
                    value={provisionForm.seed_context?.preferred_slot || ''}
                    onChange={updateSeedContextField}
                  />
                </label>
                <label>
                  Periode
                  <input
                    name="preferred_period"
                    value={provisionForm.seed_context?.preferred_period || ''}
                    onChange={updateSeedContextField}
                  />
                </label>
              </div>
            </section>

            <section className={styles.modalSection}>
              <h3>Checklist qualite</h3>
              <ul className={styles.checklist}>
                {qualityChecks.map((check) => (
                  <li key={check.label} className={check.ok ? styles.okItem : styles.warnItem}>
                    <strong>{check.ok ? 'OK' : check.warningOnly ? 'A verifier' : 'Manquant'}</strong> - {check.label}
                  </li>
                ))}
              </ul>
              <div className={styles.previewBox}>
                <h4>Apercu final</h4>
                <p>
                  <strong>Espace:</strong> {provisionForm.workspace_display_name || provisionForm.organization_name}
                </p>
                <p>
                  <strong>Persona:</strong> {provisionForm.demo_persona} | <strong>Template:</strong>{' '}
                  {provisionForm.provision_template}
                </p>
                <p>
                  <strong>Guide:</strong> {provisionForm.guide_variant}
                </p>
                <p>
                  <strong>Login demo:</strong> {provisionForm.demo_login_email}
                </p>
              </div>
            </section>

            <footer className={styles.modalFooter}>
              <button type="button" className={styles.ghostButton} onClick={closeProvisionAssistant}>
                Annuler
              </button>
              <button
                type="button"
                className={styles.primaryButton}
                disabled={busyKey === `p-${selectedRequest.id}`}
                onClick={submitProvisionAssistant}
              >
                {busyKey === `p-${selectedRequest.id}` ? 'Provisionnement...' : 'Provisionner et envoyer acces'}
              </button>
            </footer>
          </div>
        </div>
      )}
    </>
  );
};

export default AdminDemoRequests;
