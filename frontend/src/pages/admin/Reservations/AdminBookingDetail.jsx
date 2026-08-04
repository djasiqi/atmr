import React, { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';
import { fetchAdminBookingDetail } from '../../../services/adminService';
import styles from './AdminReservations.module.css';
import shell from '../adminShell.module.css';
import { adminPaths } from '../routing/adminRoutePaths';

const ROLE_LABELS = {
  company: 'Entreprise',
  driver: 'Chauffeur',
  admin: 'Administrateur',
  system: 'Système',
};

const DIAG_BADGE = {
  action_required: { label: 'Action requise', className: 'diagBadgeAction' },
  attention: { label: 'Attention', className: 'diagBadgeAttention' },
  ok: { label: 'Normal', className: 'diagBadgeOk' },
};

const RECOMMENDED_ACTION_LABELS = {
  request_or_correct_schedule: 'Confirmer ou corriger l’horaire avec le demandeur',
  request_or_correct_customer_name: 'Compléter le nom du client',
  request_or_correct_pickup: 'Compléter le lieu de départ',
  request_or_correct_dropoff: 'Compléter le lieu d’arrivée',
  retry_dispatch_or_assign: 'Relancer le dispatch ou assigner une entreprise',
  review_pending_transfer: 'Examiner le transfert en attente',
  open_investigation: 'Ouvrir une investigation technique',
  request_institution_identification: 'Identifier l’institution demandeuse',
};

const MISSION_TYPE_LABELS = {
  patient_transport: 'Transport patient',
  material_delivery: 'Livraison matériel',
};

const SOURCE_LABELS = {
  client: 'Client',
  institution_request: 'Demande institution',
  unknown: 'Non renseigné',
  booking: 'Système',
  institution: 'Institution',
  audit: 'Audit',
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

function formatRelativeAge(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return null;
  const s = Math.max(0, Number(seconds));
  if (s < 60) return `il y a ${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `il y a ${m} min`;
  const h = Math.floor(m / 60);
  if (h < 48) return `il y a ${h} h`;
  const d = Math.floor(h / 24);
  return `il y a ${d} j`;
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

function actorLabel(actor) {
  if (!actor) return null;
  return actor.label || (actor.id != null ? `#${actor.id}` : null);
}

function EmptyValue() {
  return <span className={styles.valueEmpty}>Non renseigné</span>;
}

function displayOrEmpty(value) {
  if (value == null || value === '' || value === '—') return <EmptyValue />;
  return value;
}

function formatTimelineDetails(details) {
  if (!details || typeof details !== 'object') return null;
  const entries = Object.entries(details);
  if (!entries.length) return null;
  return entries.map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(', ');
}

function resolveBackPath(location, adminId, bookingsListPath) {
  const from = location.state?.from;
  if (typeof from !== 'string' || !from.startsWith('/')) return bookingsListPath;
  const expectedPrefix = adminPaths.operationsBookings(adminId);
  if (from === expectedPrefix || from.startsWith(`${expectedPrefix}?`)) {
    return from;
  }
  return bookingsListPath;
}

function tripTypeLabel(transport) {
  if (!transport) return '—';
  if (transport.is_round_trip) return 'Aller-retour';
  if (transport.is_return) return 'Retour';
  return 'Aller simple';
}

function missionTypeLabel(raw) {
  if (!raw) return null;
  return MISSION_TYPE_LABELS[raw] || raw;
}

const AdminBookingDetail = () => {
  const { public_id: adminId, bookingId } = useParams();
  const location = useLocation();
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
  const backPath = resolveBackPath(location, adminId, bookingsListPath);

  const transport = data?.transport;
  const diagnostic = data?.support_diagnostic;
  const actors = data?.actors;
  const diagMeta = DIAG_BADGE[diagnostic?.status] || DIAG_BADGE.ok;

  const companySearchPath = useMemo(() => {
    const name = actors?.current_company?.label;
    if (!name) return null;
    const base = adminPaths.partnersUsers(adminId);
    return `${base}?search=${encodeURIComponent(name)}`;
  }, [actors, adminId]);

  const investigationPath = useMemo(() => {
    const base = adminPaths.advancedPlatform(adminId, 'investigation');
    const id = data?.references?.booking_id ?? bookingId;
    return `${base}?booking_id=${encodeURIComponent(String(id))}`;
  }, [adminId, bookingId, data?.references?.booking_id]);

  const reasons = diagnostic?.reasons || [];
  const opsReasons = reasons.filter((r) => r.severity !== 'info');
  const infoReasons = reasons.filter((r) => r.severity === 'info');
  const blockingCount = reasons.filter((r) => r.severity === 'blocking').length;
  const relativeUpdate = formatRelativeAge(transport?.last_updated_age_seconds);
  const recommendedLabel =
    RECOMMENDED_ACTION_LABELS[diagnostic?.recommended_action] || null;

  const headerMeta = useMemo(() => {
    const parts = [];
    if (transport?.status_label) parts.push(transport.status_label);
    if (actors?.current_company?.label) parts.push(actors.current_company.label);
    return parts.join(' · ');
  }, [transport, actors]);

  return (
    <main className={`${shell.content} ${styles.detailMain}`}>
      <header className={styles.detailPageHeader}>
        <Link to={backPath} className={styles.detailBack}>
          ← Retour aux transports
        </Link>

        <div className={styles.detailHero}>
          <p className={styles.detailEyebrow}>Console support</p>
          <div className={styles.detailTitleHeading}>
            <h1 className={styles.detailH1}>Transport nº {bookingId}</h1>
            {diagnostic ? (
              <span className={`${styles.diagBadge} ${styles[diagMeta.className]}`}>
                {diagMeta.label}
              </span>
            ) : null}
          </div>
          {diagnostic?.headline ? (
            <p className={styles.detailSubtitle}>{diagnostic.headline}</p>
          ) : (
            <p className={styles.detailSubtitle}>{headerMeta || 'Chargement…'}</p>
          )}
          {transport ? (
            <p className={styles.detailRouteLine}>
              <span className={styles.detailRoutePoint}>{transport.pickup || 'Départ —'}</span>
              <span className={styles.detailRouteArrow} aria-hidden>
                →
              </span>
              <span className={styles.detailRoutePoint}>{transport.dropoff || 'Arrivée —'}</span>
            </p>
          ) : null}
          {headerMeta && diagnostic?.headline ? (
            <p className={styles.detailHeaderMeta}>{headerMeta}</p>
          ) : null}
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
        <div className={styles.detailSupportLayout}>
          {diagnostic ? (
            <section
              className={`${styles.diagCard} ${styles[`diagCard_${diagnostic.status}`] || ''}`}
              aria-labelledby="detail-diagnostic"
            >
              <div className={styles.diagCardHeader}>
                <h2 id="detail-diagnostic" className={styles.diagCardTitle}>
                  Diagnostic
                </h2>
                <span className={`${styles.diagBadge} ${styles[diagMeta.className]}`}>
                  {diagMeta.label}
                </span>
              </div>
              <p className={styles.diagHeadline}>{diagnostic.headline}</p>
              <p className={styles.diagSummary}>{diagnostic.summary}</p>

              {opsReasons.length > 0 ? (
                <ul className={styles.diagReasonList}>
                  {opsReasons.map((r) => (
                    <li key={r.code} className={styles[`diagSeverity_${r.severity}`]}>
                      <span className={styles.diagReasonDot} aria-hidden />
                      <span className={styles.diagReasonLabel}>{r.label}</span>
                    </li>
                  ))}
                </ul>
              ) : null}

              {recommendedLabel ? (
                <div className={styles.diagRecommendedBox}>
                  <span className={styles.diagRecommendedLabel}>Prochaine étape</span>
                  <p className={styles.diagRecommended}>{recommendedLabel}</p>
                </div>
              ) : null}

              {infoReasons.length > 0 ? (
                <div className={styles.diagInfoBlock}>
                  <p className={styles.diagInfoTitle}>Traçabilité</p>
                  <ul className={styles.diagInfoList}>
                    {infoReasons.map((r) => (
                      <li key={r.code}>{r.label}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </section>
          ) : null}

          <div className={styles.detailSupportColumns}>
            <div className={styles.detailSupportMain}>
              <section className={styles.sectionCard} aria-labelledby="detail-transport">
                <h2 id="detail-transport" className={styles.sectionCardTitle}>
                  Transport
                </h2>
                <dl className={styles.kvList}>
                  <div className={styles.kvRow}>
                    <dt>Statut</dt>
                    <dd>
                      <span
                        className={`${styles.statusBadge} ${statusBadgeClassName(
                          transport?.status,
                          styles,
                        )}`}
                      >
                        {transport?.status_label ?? '—'}
                      </span>
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Date et heure</dt>
                    <dd>
                      {transport?.scheduled_at ? (
                        formatDateTime(transport.scheduled_at)
                      ) : (
                        <span className={styles.valueWarn}>À définir</span>
                      )}
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Dernière mise à jour</dt>
                    <dd>
                      {formatDateTime(transport?.last_updated_at)}
                      {relativeUpdate ? (
                        <span className={styles.kvMeta}> · {relativeUpdate}</span>
                      ) : null}
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Départ</dt>
                    <dd className={styles.kvMultiline}>
                      {displayOrEmpty(transport?.pickup)}
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Arrivée</dt>
                    <dd className={styles.kvMultiline}>
                      {displayOrEmpty(transport?.dropoff)}
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Type</dt>
                    <dd>
                      {tripTypeLabel(transport)}
                      {missionTypeLabel(transport?.mission_type) ? (
                        <span className={styles.kvMeta}>
                          {' '}
                          · {missionTypeLabel(transport.mission_type)}
                        </span>
                      ) : null}
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Montant</dt>
                    <dd>
                      {transport?.amount_chf != null ? (
                        <span className={styles.amountHighlight}>{transport.amount_chf} CHF</span>
                      ) : (
                        <EmptyValue />
                      )}
                    </dd>
                  </div>
                </dl>
              </section>

              <section className={styles.sectionCard} aria-labelledby="detail-actors">
                <h2 id="detail-actors" className={styles.sectionCardTitle}>
                  Acteurs
                </h2>
                <dl className={styles.kvList}>
                  <div className={styles.kvRow}>
                    <dt>Client</dt>
                    <dd>{displayOrEmpty(actorLabel(actors?.client))}</dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Institution</dt>
                    <dd>{displayOrEmpty(actorLabel(actors?.institution))}</dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Demandeur</dt>
                    <dd>
                      {actors?.requester ? (
                        <>
                          {actorLabel(actors.requester)}
                          {actors.requester.source ? (
                            <span className={styles.kvMeta}>
                              {' '}
                              · {SOURCE_LABELS[actors.requester.source] || actors.requester.source}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        <EmptyValue />
                      )}
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Entreprise actuelle</dt>
                    <dd className={styles.kvStrong}>
                      {displayOrEmpty(actorLabel(actors?.current_company))}
                    </dd>
                  </div>
                  {actors?.executing_company ? (
                    <div className={styles.kvRow}>
                      <dt>Entreprise exécutante</dt>
                      <dd>{actorLabel(actors.executing_company)}</dd>
                    </div>
                  ) : null}
                  {actors?.previous_company ? (
                    <div className={styles.kvRow}>
                      <dt>Entreprise précédente</dt>
                      <dd>{actorLabel(actors.previous_company)}</dd>
                    </div>
                  ) : null}
                  <div className={styles.kvRow}>
                    <dt>Chauffeur</dt>
                    <dd>
                      {actors?.driver ? (
                        actorLabel(actors.driver)
                      ) : (
                        <span className={styles.valueMuted}>Non affecté</span>
                      )}
                    </dd>
                  </div>
                  {actors?.cancelled_by ? (
                    <div className={styles.kvRow}>
                      <dt>Annulation</dt>
                      <dd>
                        <div className={styles.cancelledBlock}>
                          <div className={styles.cancelledLine}>
                            <span className={styles.cancelledLabel}>Acteur</span>
                            <span className={styles.kvStrong}>
                              {ROLE_LABELS[actors.cancelled_by.role] ??
                                actors.cancelled_by.role ??
                                '—'}
                            </span>
                          </div>
                          {actors.cancelled_by.cancelled_at ? (
                            <div className={styles.cancelledLine}>
                              <span className={styles.cancelledLabel}>Date</span>
                              <span>{formatDateTime(actors.cancelled_by.cancelled_at)}</span>
                            </div>
                          ) : null}
                          {actors.cancelled_by.reason_code ? (
                            <div className={styles.cancelledLine}>
                              <span className={styles.cancelledLabel}>Motif</span>
                              <code className={styles.inlineCode}>
                                {actors.cancelled_by.reason_code}
                              </code>
                            </div>
                          ) : null}
                        </div>
                      </dd>
                    </div>
                  ) : null}
                </dl>
              </section>

              <section className={styles.sectionCard} aria-labelledby="detail-timeline">
                <h2 id="detail-timeline" className={styles.sectionCardTitle}>
                  Chronologie
                </h2>
                <ul className={styles.timelineRail}>
                  {(data.timeline || []).length === 0 ? (
                    <li className={styles.timelineEmpty}>Aucun événement enregistré.</li>
                  ) : (
                    (data.timeline || []).map((ev, idx) => {
                      const detailText =
                        ev.detail ||
                        formatTimelineDetails(ev.details) ||
                        (ROLE_LABELS[ev.actor] ? ROLE_LABELS[ev.actor] : null);
                      const sourceLabel = SOURCE_LABELS[ev.source] || null;
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
                              {sourceLabel ? (
                                <span className={styles.timelineSource}>{sourceLabel}</span>
                              ) : null}
                            </div>
                          </div>
                        </li>
                      );
                    })
                  )}
                </ul>
              </section>
            </div>

            <aside className={styles.detailSupportRail}>
              <section className={styles.railCard} aria-labelledby="detail-state">
                <h2 id="detail-state" className={styles.railCardTitle}>
                  Synthèse
                </h2>
                <ul className={styles.railStatusList}>
                  <li>
                    <span className={styles.railStatusKey}>Données</span>
                    <span
                      className={
                        blockingCount > 0 ? styles.railStatusBad : styles.railStatusOk
                      }
                    >
                      {blockingCount > 0
                        ? `${blockingCount} anomalie${blockingCount > 1 ? 's' : ''}`
                        : 'OK'}
                    </span>
                  </li>
                  <li>
                    <span className={styles.railStatusKey}>Chauffeur</span>
                    <span className={styles.railStatusVal}>
                      {actors?.driver ? actorLabel(actors.driver) : 'Non affecté'}
                    </span>
                  </li>
                  <li>
                    <span className={styles.railStatusKey}>À investiguer</span>
                    <span
                      className={
                        diagnostic?.needs_investigation
                          ? styles.railStatusBad
                          : styles.railStatusOk
                      }
                    >
                      {diagnostic?.needs_investigation ? 'Oui' : 'Non'}
                    </span>
                  </li>
                </ul>
              </section>

              <section className={styles.railCard} aria-labelledby="detail-links">
                <h2 id="detail-links" className={styles.railCardTitle}>
                  Accès rapides
                </h2>
                <nav className={styles.railLinkNav} aria-label="Raccourcis administration">
                  {companySearchPath ? (
                    <Link className={styles.railLink} to={companySearchPath}>
                      Entreprise {actors?.current_company?.label || ''}
                    </Link>
                  ) : null}
                  <Link className={styles.railLink} to={investigationPath}>
                    Investigation technique
                  </Link>
                  <Link className={styles.railLink} to={bookingsListPath}>
                    Liste des transports
                  </Link>
                </nav>
              </section>

              <details className={styles.techDetails}>
                <summary>Détails techniques</summary>
                <dl className={`${styles.kvList} ${styles.techKv}`}>
                  <div className={styles.kvRow}>
                    <dt>ID transport</dt>
                    <dd>
                      <code className={styles.inlineCode}>
                        {data.references?.booking_id ?? bookingId}
                      </code>
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Code statut</dt>
                    <dd>
                      <code className={styles.inlineCode}>{transport?.status ?? '—'}</code>
                    </dd>
                  </div>
                  <div className={styles.kvRow}>
                    <dt>Version</dt>
                    <dd>
                      <code className={styles.inlineCode}>
                        {transport?.edit_version ?? '—'}
                      </code>
                    </dd>
                  </div>
                  {actors?.current_company?.id != null ? (
                    <div className={styles.kvRow}>
                      <dt>ID entreprise</dt>
                      <dd>
                        <code className={styles.inlineCode}>{actors.current_company.id}</code>
                      </dd>
                    </div>
                  ) : null}
                  {actors?.driver?.id != null ? (
                    <div className={styles.kvRow}>
                      <dt>ID chauffeur</dt>
                      <dd>
                        <code className={styles.inlineCode}>{actors.driver.id}</code>
                      </dd>
                    </div>
                  ) : null}
                  {diagnostic?.recommended_action ? (
                    <div className={styles.kvRow}>
                      <dt>Action code</dt>
                      <dd>
                        <code className={styles.inlineCode}>
                          {diagnostic.recommended_action}
                        </code>
                      </dd>
                    </div>
                  ) : null}
                </dl>
              </details>
            </aside>
          </div>
        </div>
      )}
    </main>
  );
};

export default AdminBookingDetail;

export { resolveBackPath, formatRelativeAge };
