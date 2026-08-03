import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiX } from 'react-icons/fi';
import { fetchAccountIntegrity } from '../../../services/adminService';
import styles from './AdminAccountIntegrityDrawer.module.css';

const formatDate = (value) => {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString('fr-CH');
};

const statusIcon = (status) => {
  if (status === 'passed') return '✓';
  if (status === 'failed') return '✕';
  if (status === 'warning') return '!';
  return '—';
};

/**
 * Diagnostic d'intégrité pour un compte orphelin / incohérent (lecture seule).
 */
const AdminAccountIntegrityDrawer = ({ accountId, isOpen, onClose }) => {
  const overlayRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [payload, setPayload] = useState(null);

  useEffect(() => {
    if (!isOpen || !accountId) return undefined;
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchAccountIntegrity(accountId);
        if (!cancelled) setPayload(data);
      } catch (err) {
        if (!cancelled) {
          setError(err?.response?.data?.message || 'Impossible de charger le diagnostic.');
          setPayload(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [isOpen, accountId]);

  useEffect(() => {
    if (!isOpen) return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, [isOpen]);

  if (!isOpen || !accountId) return null;

  const account = payload?.account || {};
  const deps = payload?.dependencies || {};
  const checks = payload?.checks || [];

  const drawer = (
    <div
      ref={overlayRef}
      className={styles.overlay}
      role="presentation"
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <aside className={styles.drawer} role="dialog" aria-modal="true" aria-labelledby="integrity-title">
        <header className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Diagnostic</p>
            <h2 id="integrity-title" className={styles.title}>
              {account.username || account.email || `Compte #${accountId}`}
            </h2>
            <p className={styles.subtitle}>
              {payload?.configuration_status === 'incomplete'
                ? 'Configuration incomplète'
                : 'Configuration'}
            </p>
          </div>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={18} aria-hidden />
          </button>
        </header>

        <div className={styles.body}>
          {loading ? <p className={styles.muted}>Chargement du diagnostic…</p> : null}
          {error ? (
            <p className={styles.error} role="alert">
              {error}
            </p>
          ) : null}

          {payload ? (
            <>
              <p className={styles.lead}>
                {account.role === 'COMPANY'
                  ? 'Le compte possède le rôle Entreprise, mais aucune fiche entreprise ne lui est liée.'
                  : account.role === 'INSTITUTION'
                    ? 'Le compte possède le rôle Institution, mais aucune institution ne lui est liée.'
                    : 'Diagnostic du compte.'}
              </p>
              <p className={styles.notice}>
                Aucune modification n&apos;est disponible depuis cette page.
              </p>

              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Compte</h3>
                <dl className={styles.kv}>
                  <div>
                    <dt>E-mail</dt>
                    <dd>{account.email || '—'}</dd>
                  </div>
                  <div>
                    <dt>Créé le</dt>
                    <dd>{formatDate(account.created_at)}</dd>
                  </div>
                  <div>
                    <dt>Rôle</dt>
                    <dd>{account.role || '—'}</dd>
                  </div>
                </dl>
              </section>

              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Vérifications</h3>
                <ul className={styles.checks}>
                  {checks.map((check) => (
                    <li key={check.code}>
                      <span aria-hidden>{statusIcon(check.status)}</span> {check.label}
                    </li>
                  ))}
                </ul>
              </section>

              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Dépendances</h3>
                <p className={styles.muted}>
                  {deps.driver_profile_exists ? '1 profil chauffeur' : '0 chauffeur'}
                  {' · '}
                  {deps.bookings_created_by_account_count || 0} transport(s) créés
                  {' · '}
                  {deps.client_profiles_count || 0} profil(s) client
                  {' · '}
                  {deps.refresh_sessions_count || 0} session(s)
                </p>
              </section>

              {payload.possible_matches?.length ? (
                <section className={styles.section}>
                  <h3 className={styles.sectionTitle}>Correspondances possibles</h3>
                  <ul className={styles.checks}>
                    {payload.possible_matches.map((m) => (
                      <li key={`${m.organization_key}-${m.reason}`}>
                        {m.organization_key} ({m.confidence})
                      </li>
                    ))}
                  </ul>
                </section>
              ) : null}

              {payload.recommendation ? (
                <section className={styles.section}>
                  <h3 className={styles.sectionTitle}>Recommandation</h3>
                  <p className={styles.muted}>{payload.recommendation}</p>
                </section>
              ) : null}
            </>
          ) : null}
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

export default AdminAccountIntegrityDrawer;
