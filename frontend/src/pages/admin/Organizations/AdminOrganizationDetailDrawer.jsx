import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import styles from './AdminOrganizationDetailDrawer.module.css';

const TYPE_LABELS = {
  company: 'Entreprise de transport',
  institution: 'Institution',
};

const formatDate = (value) => {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString('fr-CH');
};

/**
 * Détail organisation (lecture seule) — Company ou Institution.
 */
const AdminOrganizationDetailDrawer = ({ organization, isOpen, onClose }) => {
  const drawerRef = useRef(null);
  const overlayRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, [isOpen]);

  if (!isOpen || !organization) return null;

  const isCompany = organization.organization_type === 'company';
  const lifecycle =
    organization.lifecycle_status === 'suspended'
      ? 'Suspendu'
      : organization.lifecycle_status === 'active'
        ? 'Actif'
        : 'État inconnu';

  const drawer = (
    <div
      ref={overlayRef}
      className={styles.overlay}
      role="presentation"
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <aside
        ref={drawerRef}
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="org-drawer-title"
      >
        <header className={styles.header}>
          <div>
            <p className={styles.eyebrow}>
              {TYPE_LABELS[organization.organization_type] || organization.organization_type}
            </p>
            <h2 id="org-drawer-title" className={styles.title}>
              {organization.name || 'Organisation'}
            </h2>
            <p className={styles.subtitle}>
              {lifecycle}
              {organization.configuration_status === 'incomplete' ? ' · À compléter' : ''}
            </p>
          </div>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={18} aria-hidden />
          </button>
        </header>

        <div className={styles.body}>
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Configuration</h3>
            <ul className={styles.checklist}>
              <li>
                Fiche organisation{' '}
                {organization.configuration_status === 'complete' ? '✓' : '✕'}
              </li>
              <li>
                Compte principal{' '}
                {organization.primary_account ? '✓' : '—'}
              </li>
              <li>
                Utilisateurs : {organization.accounts_count ?? '—'}
              </li>
              <li>
                Administrateurs : {organization.administrators_count ?? '—'}
              </li>
              {isCompany ? (
                <li>Chauffeurs : {organization.drivers_count ?? 0}</li>
              ) : null}
              {isCompany ? (
                <li>
                  Accès commercial : {organization.commercial_access_state || 'active'}
                </li>
              ) : null}
            </ul>
          </section>

          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Organisation</h3>
            <dl className={styles.kv}>
              <div>
                <dt>Contact</dt>
                <dd>{organization.contact_email || '—'}</dd>
              </div>
              <div>
                <dt>Portée des données</dt>
                <dd>
                  {organization.data_scope}
                  {organization.contains_synthetic_accounts
                    ? ' (contient des comptes synthétiques)'
                    : ''}
                </dd>
              </div>
              {organization.primary_account ? (
                <div>
                  <dt>Compte principal</dt>
                  <dd>
                    {organization.primary_account.name || '—'}
                    <br />
                    {organization.primary_account.email || ''}
                  </dd>
                </div>
              ) : null}
            </dl>
          </section>

          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Historique</h3>
            <p className={styles.muted}>
              Organisation créée le {formatDate(organization.created_at)}.
            </p>
          </section>
        </div>

        <footer className={styles.footer}>
          <button type="button" className={styles.footerClose} onClick={onClose}>
            Fermer
          </button>
        </footer>
      </aside>
    </div>
  );

  return createPortal(drawer, document.body);
};

export default AdminOrganizationDetailDrawer;
