import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { FaClipboardList } from 'react-icons/fa';
import { fetchAdminBookingDetail } from '../../../services/adminService';
import styles from './AdminReservations.module.css';
import shell from '../adminShell.module.css';
import { adminPaths } from '../routing/adminRoutePaths';

const SOURCE_LABELS = {
  client: 'Client',
  institution_request: 'Demande institution',
  unknown: 'Non renseigné',
};

const ROLE_LABELS = {
  company: 'Entreprise',
  driver: 'Chauffeur',
  admin: 'Administrateur',
  system: 'Système',
};

function formatDateTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('fr-CH', {
      dateStyle: 'medium',
      timeStyle: 'medium',
    });
  } catch {
    return String(iso);
  }
}

function statusBadgeClassName(status, stylesObj) {
  const s = String(status || '').toLowerCase();
  if (s === 'canceled' || s === 'cancelled') return stylesObj.statusCancelled;
  if (s === 'pending') return stylesObj.statusPending;
  if (s === 'completed' || s === 'return_completed') return stylesObj.statusCompleted;
  if (s === 'assigned' || s === 'accepted') return stylesObj.statusAssigned;
  if (s === 'rejected') return stylesObj.statusRejected;
  return stylesObj.statusDefault;
}

function translateTimelineDetail(detail) {
  if (detail == null || detail === '') return null;
  const d = String(detail);
  if (ROLE_LABELS[d]) return ROLE_LABELS[d];
  return d;
}

const AdminBookingDetail = () => {
  const { public_id: adminId, bookingId } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const payload = await fetchAdminBookingDetail(bookingId);
        if (!cancelled) setData(payload);
      } catch (err) {
        if (!cancelled) {
          setError(err?.response?.data?.message || err?.message || 'Erreur');
          setData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [bookingId]);

  const bookingsListPath = adminPaths.operationsBookings(adminId);
  const booking = data?.booking;

  const createdBy = data?.created_by;
  const cancelledBy = data?.cancelled_by;
  const sourceKey = createdBy?.source && SOURCE_LABELS[createdBy.source] ? createdBy.source : 'unknown';

  return (
    <main className={`${shell.content} ${styles.detailMain}`}>
      <header className={styles.detailPageHeader}>
        <Link to={bookingsListPath} className={styles.detailBack}>
          ← Retour aux réservations
        </Link>
        <div className={styles.detailTitleRow}>
          <span className={styles.detailTitleIcon} aria-hidden>
            <FaClipboardList />
          </span>
          <div className={styles.detailTitleText}>
            <h1 className={styles.detailH1}>Réservation #{bookingId}</h1>
            <p className={styles.detailSubtitle}>
              Supervision plateforme — libellés et statuts issus de l&apos;API.
            </p>
          </div>
        </div>
      </header>

      {loading && (
        <div className={styles.detailLoading} role="status">
          Chargement des données…
        </div>
      )}
      {error && (
        <div className={styles.detailErrorBox} role="alert">
          <strong>Impossible de charger le détail.</strong> {error}
        </div>
      )}

      {!loading && !error && data && (
        <div className={styles.detailLayout}>
          <section className={styles.sectionCard} aria-labelledby="detail-identity">
            <h2 id="detail-identity" className={styles.sectionCardTitle}>
              Identité
            </h2>
            <p className={styles.sectionCardLead}>Client, structure et responsabilité opérationnelle.</p>
            <dl className={styles.kvList}>
              <div className={styles.kvRow}>
                <dt>Client</dt>
                <dd>{data.client_name ?? '—'}</dd>
              </div>
              <div className={styles.kvRow}>
                <dt>Institution</dt>
                <dd>{data.institution_name ?? '—'}</dd>
              </div>
              <div className={styles.kvRow}>
                <dt>Statut</dt>
                <dd className={styles.kvRowStatus}>
                  <span
                    className={`${styles.statusBadge} ${statusBadgeClassName(data.status, styles)}`}
                  >
                    {data.status_label ?? data.status}
                  </span>
                  <code className={styles.inlineCode}>{data.status}</code>
                </dd>
              </div>
              <div className={styles.kvRow}>
                <dt>Entreprise actuelle</dt>
                <dd>
                  {data.current_company ? (
                    <>
                      <span className={styles.kvStrong}>{data.current_company.name}</span>
                      <span className={styles.kvMeta}>
                        {' '}
                        · id <code className={styles.inlineCode}>{data.current_company.id}</code>
                      </span>
                    </>
                  ) : (
                    '—'
                  )}
                </dd>
              </div>
              {data.previous_company ? (
                <div className={styles.kvRow}>
                  <dt>Entreprise précédente</dt>
                  <dd>
                    <span className={styles.kvStrong}>{data.previous_company.name}</span>
                    <span className={styles.kvMeta}>
                      {' '}
                      · transfert · id{' '}
                      <code className={styles.inlineCode}>{data.previous_company.id}</code>
                    </span>
                  </dd>
                </div>
              ) : null}
              <div className={styles.kvRow}>
                <dt>Créée par</dt>
                <dd>
                  {createdBy ? (
                    <div className={styles.createdByBlock}>
                      <span className={styles.sourcePill}>
                        {SOURCE_LABELS[sourceKey] || createdBy.source || 'Autre'}
                      </span>

                      <div className={styles.createdByName}>{createdBy.label ?? '—'}</div>
                      {createdBy.institution_name ? (
                        <div className={styles.createdByMeta}>{createdBy.institution_name}</div>
                      ) : null}
                    </div>
                  ) : (
                    '—'
                  )}
                </dd>
              </div>
              <div className={styles.kvRow}>
                <dt>Annulation</dt>
                <dd>
                  {cancelledBy ? (
                    <div className={styles.cancelledBlock}>
                      <div className={styles.cancelledLine}>
                        <span className={styles.cancelledLabel}>Acteur</span>
                        <span className={styles.kvStrong}>
                          {ROLE_LABELS[cancelledBy.role] ?? cancelledBy.role ?? '—'}
                        </span>
                      </div>
                      {cancelledBy.cancelled_at ? (
                        <div className={styles.cancelledLine}>
                          <span className={styles.cancelledLabel}>Date</span>
                          <span>{formatDateTime(cancelledBy.cancelled_at)}</span>
                        </div>
                      ) : null}
                      {cancelledBy.reason_code ? (
                        <div className={styles.cancelledLine}>
                          <span className={styles.cancelledLabel}>Motif</span>
                          <code className={styles.inlineCode}>{cancelledBy.reason_code}</code>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <span className={styles.subtle}>Non applicable</span>
                  )}
                </dd>
              </div>
            </dl>
          </section>

          {booking ? (
            <section className={styles.sectionCard} aria-labelledby="detail-route">
              <h2 id="detail-route" className={styles.sectionCardTitle}>
                Trajet
              </h2>
              <p className={styles.sectionCardLead}>Extrait des lieux et du montant affiché côté mission.</p>
              <dl className={styles.kvList}>
                <div className={styles.kvRow}>
                  <dt>Départ</dt>
                  <dd className={styles.kvMultiline}>{booking.pickup_location ?? '—'}</dd>
                </div>
                <div className={styles.kvRow}>
                  <dt>Arrivée</dt>
                  <dd className={styles.kvMultiline}>{booking.dropoff_location ?? '—'}</dd>
                </div>
                <div className={styles.kvRow}>
                  <dt>Montant</dt>
                  <dd>
                    {booking.amount != null ? (
                      <span className={styles.amountHighlight}>{booking.amount} CHF</span>
                    ) : (
                      '—'
                    )}
                  </dd>
                </div>
              </dl>
            </section>
          ) : null}

          <section className={styles.sectionCard} aria-labelledby="detail-timeline">
            <h2 id="detail-timeline" className={styles.sectionCardTitle}>
              Chronologie
            </h2>
            <p className={styles.sectionCardLead}>
              Jalons métier et événements d&apos;audit associés à cette réservation.
            </p>
            <ul className={styles.timelineRail}>
              {(data.timeline || []).map((ev, idx) => {
                const detailText = translateTimelineDetail(ev.detail);
                return (
                  <li key={`${ev.type}-${ev.at}-${idx}`}>
                    <div className={styles.timelineCard}>
                      <time className={styles.timelineTime} dateTime={ev.at || undefined}>
                        {ev.at ? formatDateTime(ev.at) : '—'}
                      </time>
                      <div className={styles.timelineBody}>
                        <span className={styles.timelineLabel}>{ev.label || ev.type}</span>
                        {detailText ? (
                          <span className={styles.timelineDetail}>{detailText}</span>
                        ) : null}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          </section>

          <section className={styles.sectionCard} aria-labelledby="detail-links">
            <h2 id="detail-links" className={styles.sectionCardTitle}>
              Raccourcis
            </h2>
            <p className={styles.sectionCardLead}>Navigation vers les écrans admin et la console plateforme.</p>
            <nav className={styles.linkNav} aria-label="Raccourcis administration">
              {data.links?.platform_ops ? (
                <Link className={styles.linkPill} to={data.links.platform_ops}>
                  Console plateforme
                </Link>
              ) : null}
              {data.links?.institution ? (
                <Link className={styles.linkPill} to={data.links.institution}>
                  Contexte institution
                </Link>
              ) : null}
              {data.links?.company ? (
                <Link className={styles.linkPill} to={data.links.company}>
                  Utilisateurs / entreprises
                </Link>
              ) : null}
              <Link className={styles.linkPill} to={bookingsListPath}>
                Tableau de bord admin
              </Link>
            </nav>
          </section>
        </div>
      )}
    </main>
  );
};

export default AdminBookingDetail;
