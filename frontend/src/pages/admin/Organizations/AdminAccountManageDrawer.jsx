import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link, useParams } from 'react-router-dom';
import { FiX } from 'react-icons/fi';
import {
  fetchAccountManageContext,
  fetchCompanyDispatchDisablePreview,
  pauseCompanyDunning,
  previewUserRoleTransition,
  resetUserPassword,
  resumeCompanyDunning,
  revokeUserSessions,
  setCompanyApproval,
  setCompanyBillingAccess,
  setCompanyDispatchStatus,
  updateDriverStatus,
  updateUserRole,
} from '../../../services/adminService';
import { useAdminCapabilities } from '../../../hooks/useAdminCapabilities';
import { adminPaths } from '../routing/adminRoutePaths';
import AdminActionDialog from '../components/AdminActionDialog';
import AdminTempPasswordDialog from '../components/AdminTempPasswordDialog';
import styles from './AdminAccountManageDrawer.module.css';

const ROLE_OPTIONS = [
  { value: 'client', label: 'Client' },
  { value: 'driver', label: 'Chauffeur' },
  { value: 'company', label: 'Entreprise' },
  { value: 'institution', label: 'Institution' },
  { value: 'admin', label: 'Admin' },
];

const RESTRICTION_LABELS = {
  active: 'Aucune restriction',
  partial: 'Restriction partielle',
  full: 'Restriction complète',
};

const RESTRICTION_IMPACT = {
  active:
    "L'entreprise peut accéder normalement au portefeuille propre, à la marketplace et aux services facturables.",
  partial:
    'La marketplace et certains services facturables seront restreints. Les courses déjà créées, le GPS, les factures et la connexion restent disponibles.',
  full:
    'En plus des restrictions partielles, l’entreprise ne pourra plus créer de nouvelles courses pour son portefeuille propre.',
};

const INSTITUTION_ROLE_LABELS = {
  institution_admin: 'Administrateur',
  institution_requester: 'Demandeur',
  institution_reader: 'Lecteur',
  institution_billing: 'Facturation',
  institution_curator: 'Curateur',
  institution_reception: 'Réception',
};

const formatDate = (value) => {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString('fr-CH');
};

const formatRestrictionLabel = (state) =>
  RESTRICTION_LABELS[String(state || 'active').toLowerCase()] || 'Aucune restriction';

const restrictionBadgeClass = (state) => {
  const key = String(state || 'active').toLowerCase();
  if (key === 'partial') return styles.badgePartial;
  if (key === 'full') return styles.badgeFull;
  return styles.badgeActive;
};

const checkClass = (status) => {
  if (status === 'passed') return styles.checkOk;
  if (status === 'failed') return styles.checkFail;
  if (status === 'warning') return styles.checkWarn;
  return styles.checkNeutral;
};

const defaultPauseUntilLocal = () => {
  const d = new Date();
  d.setDate(d.getDate() + 14);
  d.setHours(23, 59, 0, 0);
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

/**
 * Fiche compte admin — gestion (MDP, rôle, restriction commerciale LIRIE) + diagnostic.
 */
const AdminAccountManageDrawer = ({ accountId, isOpen, onClose, onChanged }) => {
  const overlayRef = useRef(null);
  const { public_id: publicId } = useParams();
  const { canUsersManage, canUsersSecurity, canBillingLock } = useAdminCapabilities();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ctx, setCtx] = useState(null);
  const [roleDraft, setRoleDraft] = useState('client');
  const [companyId, setCompanyId] = useState('');
  const [institutionId, setInstitutionId] = useState('');
  const [institutionRole, setInstitutionRole] = useState('institution_admin');
  const [actionDialog, setActionDialog] = useState(null);
  const [tempPasswordDialog, setTempPasswordDialog] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dunningUntilLocal, setDunningUntilLocal] = useState(defaultPauseUntilLocal);
  const reasonId = useId();

  const reload = async () => {
    if (!accountId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAccountManageContext(accountId);
      setCtx(data);
      const role = String(data?.account?.role || 'CLIENT').toLowerCase();
      setRoleDraft(role);
      setCompanyId(
        data?.legacy_context?.company_id != null
          ? String(data.legacy_context.company_id)
          : ''
      );
      setInstitutionId(
        data?.legacy_context?.institution_id != null
          ? String(data.legacy_context.institution_id)
          : ''
      );
      setInstitutionRole(
        data?.legacy_context?.institution_role || 'institution_admin'
      );
    } catch (err) {
      setError(err?.response?.data?.message || 'Impossible de charger le compte.');
      setCtx(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isOpen || !accountId) return undefined;
    let cancelled = false;
    (async () => {
      if (!cancelled) await reload();
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const account = ctx?.account || {};
  const legacy = ctx?.legacy_context || {};
  const allowed = ctx?.allowed_actions || {};
  const options = ctx?.role_transition_options || {};
  const companyProfile = ctx?.company_profile;
  const commercialRestriction =
    ctx?.commercial_restriction ||
    (ctx?.commercial_access
      ? {
          company_id: ctx.commercial_access.company_id,
          state: ctx.commercial_access.platform_billing_access_state,
          dunning_paused_until: ctx.commercial_access.dunning_paused_until,
        }
      : null);
  const detectedServices = ctx?.detected_services;
  const driverProfile = ctx?.driver_profile;
  const diagnostic = ctx?.diagnostic || {};
  const isCompanyRole = String(account.role || '').toUpperCase() === 'COMPANY';
  const showCompanySupport = Boolean(companyProfile) && isCompanyRole;
  const showCommercial = showCompanySupport && Boolean(commercialRestriction);

  const canReset = Boolean(
    allowed.reset_password != null ? allowed.reset_password : canUsersSecurity
  );
  const canRevoke = Boolean(
    allowed.revoke_sessions != null ? allowed.revoke_sessions : canUsersSecurity
  );
  const canRole = Boolean(
    allowed.change_role != null ? allowed.change_role : canUsersManage
  );
  const canDriverStatus = Boolean(allowed.change_driver_status);
  const canCommercial = Boolean(
    allowed.manage_commercial_restriction != null
      ? allowed.manage_commercial_restriction
      : allowed.manage_billing_access != null
        ? allowed.manage_billing_access
        : canBillingLock && showCommercial
  );
  const canDunning = Boolean(
    allowed.pause_dunning != null ? allowed.pause_dunning : canCommercial
  );
  const canOpsFlags = Boolean(allowed.manage_operational_flags);
  const passwordTemporary = Boolean(
    account.force_password_change || ctx?.security?.password_temporary
  );
  const restrictionState = String(commercialRestriction?.state || 'active').toLowerCase();
  const dunningPausedUntil = commercialRestriction?.dunning_paused_until;
  const dunningPaused =
    Boolean(dunningPausedUntil) && new Date(dunningPausedUntil).getTime() > Date.now();

  const financeConfigPath =
    publicId && allowed.open_billing_configuration !== false
      ? adminPaths.financeConfig(publicId)
      : null;
  const platformOpsPath =
    publicId && allowed.open_platform_operations !== false
      ? adminPaths.advancedPlatform(publicId, 'tenants')
      : null;
  const driversListPath =
    publicId && companyProfile?.company_id
      ? `${adminPaths.partnersUsers(publicId)}?role=driver&company_id=${companyProfile.company_id}`
      : null;

  const buildExpectedPayload = () => ({
    expected_current_role: String(account.role || '').toLowerCase(),
    expected_company_id: legacy.company_id || undefined,
    expected_institution_id: legacy.institution_id || undefined,
    expected_institution_role: legacy.institution_role || undefined,
  });

  const handleResetPassword = () => {
    setActionDialog({
      kind: 'reset-password',
      title: 'Réinitialiser le mot de passe',
      description: (
        <>
          Générer un mot de passe temporaire pour « {account.username || account.email} ».
          Transmettez-le ensuite par un canal sûr. Les sessions seront révoquées.
        </>
      ),
      confirmationLabel: 'Réinitialiser',
      reason: { required: true, label: 'Motif', minLength: 5 },
      onConfirm: async ({ reason }) => {
        const response = await resetUserPassword(accountId, { reason });
        if (!response?.new_password) {
          throw new Error('Aucun mot de passe généré par le serveur.');
        }
        setActionDialog(null);
        setTempPasswordDialog({
          accountLabel: account.email || account.username || `ID ${accountId}`,
          temporaryPassword: response.new_password,
        });
        await reload();
        onChanged?.();
      },
    });
  };

  const handleApplyRole = async () => {
    const base = {
      role: roleDraft,
      company_id:
        roleDraft === 'driver' || roleDraft === 'company'
          ? Number(companyId) || undefined
          : undefined,
      institution_id:
        roleDraft === 'institution' ? Number(institutionId) || undefined : undefined,
      institution_role: roleDraft === 'institution' ? institutionRole : undefined,
      ...buildExpectedPayload(),
    };
    if (roleDraft === 'driver' && !base.company_id) {
      setError('Sélectionnez une entreprise de transport pour le chauffeur.');
      return;
    }
    if (roleDraft === 'institution' && (!base.institution_id || !base.institution_role)) {
      setError('Institution et rôle institutionnels requis.');
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const preview = await previewUserRoleTransition(accountId, base);
      if (!preview.allowed) {
        const blocker = preview.blockers?.[0];
        setError(blocker?.message || 'Transition non autorisée.');
        setBusy(false);
        return;
      }
      const needsStrongConfirm = roleDraft === 'admin';
      setActionDialog({
        kind: 'role',
        title: preview.changes?.[0]?.startsWith('Transférer')
          ? 'Transférer le chauffeur'
          : 'Changer le rôle',
        description: (
          <ul>
            {(preview.changes || []).map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        ),
        impact: (preview.warnings || []).length ? (
          <ul>
            {preview.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        ) : null,
        confirmationLabel: 'Confirmer',
        confirmText: needsStrongConfirm ? 'PROMOUVOIR' : undefined,
        reason: { required: true, label: 'Motif', minLength: 5 },
        onConfirm: async ({ reason }) => {
          await updateUserRole(accountId, {
            ...base,
            reason,
            preview_id: preview.preview_id,
          });
          setActionDialog(null);
          await reload();
          onChanged?.();
        },
      });
    } catch (err) {
      setError(
        err?.response?.data?.message || err.message || 'Échec de la prévisualisation.'
      );
    } finally {
      setBusy(false);
    }
  };

  const handleCommercialRestriction = (state) => {
    if (!commercialRestriction?.company_id && !companyProfile?.company_id) return;
    const cid = commercialRestriction?.company_id || companyProfile.company_id;
    const fromLabel = formatRestrictionLabel(restrictionState);
    const toLabel = formatRestrictionLabel(state);
    setActionDialog({
      kind: 'commercial',
      title: 'Restriction commerciale LIRIE',
      description: (
        <>
          Passer de « {fromLabel} » à « {toLabel} » pour «{' '}
          {companyProfile?.name || 'cette entreprise'} ».
        </>
      ),
      impact: <p>{RESTRICTION_IMPACT[state] || RESTRICTION_IMPACT.active}</p>,
      confirmationLabel: 'Appliquer',
      reason: { required: true, label: 'Motif', minLength: 5 },
      onConfirm: async ({ reason }) => {
        await setCompanyBillingAccess(cid, {
          state,
          reason_code: reason,
        });
        setActionDialog(null);
        await reload();
        onChanged?.();
      },
    });
  };

  const handlePauseDunning = ({ resume = false } = {}) => {
    const cid = companyProfile?.company_id || commercialRestriction?.company_id;
    if (!cid) return;
    if (resume) {
      setActionDialog({
        kind: 'dunning-resume',
        title: 'Reprendre le recouvrement',
        description:
          'Le recouvrement automatique LIRIE reprend immédiatement. Ceci ne modifie pas la restriction commerciale.',
        confirmationLabel: 'Reprendre maintenant',
        reason: { required: true, label: 'Motif', minLength: 5 },
        onConfirm: async ({ reason }) => {
          await resumeCompanyDunning(cid, { reason });
          setActionDialog(null);
          await reload();
          onChanged?.();
        },
      });
      return;
    }
    setActionDialog({
      kind: 'dunning-pause',
      title: dunningPaused ? 'Modifier la pause' : 'Mettre en pause le recouvrement',
      description: (
        <>
          <p>
            Pause du recouvrement automatique uniquement — distincte de la restriction
            commerciale.
          </p>
          <label className={styles.fieldLabel} htmlFor={`${reasonId}-dunning-until`}>
            Pause jusqu&apos;au
          </label>
          <input
            id={`${reasonId}-dunning-until`}
            type="datetime-local"
            className={styles.datetimeInput}
            defaultValue={dunningUntilLocal}
            onChange={(e) => setDunningUntilLocal(e.target.value)}
          />
        </>
      ),
      confirmationLabel: 'Enregistrer la pause',
      reason: { required: true, label: 'Motif', minLength: 5 },
      onConfirm: async ({ reason }) => {
        const local = dunningUntilLocal || defaultPauseUntilLocal();
        const pausedUntil = new Date(local).toISOString();
        await pauseCompanyDunning(cid, {
          paused_until: pausedUntil,
          reason,
        });
        setActionDialog(null);
        await reload();
        onChanged?.();
      },
    });
  };

  const handleApproval = (nextApproved) => {
    if (!companyProfile?.company_id) return;
    setActionDialog({
      kind: 'approval',
      title: nextApproved ? 'Approuver l’entreprise' : 'Retirer l’approbation',
      description: nextApproved
        ? 'L’entreprise sera validée par LIRIE. Le dispatch n’est pas activé automatiquement.'
        : 'L’entreprise ne sera plus considérée comme approuvée. Le dispatch et la restriction commerciale restent inchangés.',
      confirmationLabel: nextApproved ? 'Approuver' : 'Retirer l’approbation',
      reason: { required: true, label: 'Motif', minLength: 5 },
      onConfirm: async ({ reason }) => {
        await setCompanyApproval(companyProfile.company_id, {
          is_approved: nextApproved,
          expected_is_approved: Boolean(companyProfile.is_approved),
          reason,
        });
        setActionDialog(null);
        await reload();
        onChanged?.();
      },
    });
  };

  const handleDispatch = async (nextEnabled) => {
    if (!companyProfile?.company_id) return;
    let impact = (
      <p>
        L’approbation, la suspension plateforme et la restriction commerciale ne sont
        pas modifiées.
      </p>
    );
    if (!nextEnabled) {
      try {
        const preview = await fetchCompanyDispatchDisablePreview(
          companyProfile.company_id
        );
        impact = (
          <ul>
            <li>
              Chauffeurs actifs : {preview.active_drivers_count} /{' '}
              {preview.total_drivers_count}
            </li>
            <li>Courses actives : {preview.active_bookings_count}</li>
            {(preview.warnings || []).map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        );
      } catch {
        /* preview optionnel */
      }
    }
    setActionDialog({
      kind: 'dispatch',
      title: nextEnabled ? 'Activer le dispatch' : 'Désactiver le dispatch',
      description: nextEnabled
        ? 'Rétablir l’accès aux fonctions d’exploitation et d’affectation.'
        : 'Arrêt de l’exploitation dispatch. Les chauffeurs ne sont pas désactivés.',
      impact,
      confirmationLabel: nextEnabled ? 'Activer' : 'Désactiver',
      reason: { required: true, label: 'Motif', minLength: 5 },
      onConfirm: async ({ reason }) => {
        await setCompanyDispatchStatus(companyProfile.company_id, {
          dispatch_enabled: nextEnabled,
          expected_dispatch_enabled: Boolean(companyProfile.dispatch_enabled),
          reason,
        });
        setActionDialog(null);
        await reload();
        onChanged?.();
      },
    });
  };

  const handleRevokeSessions = () => {
    setActionDialog({
      kind: 'revoke-sessions',
      title: 'Révoquer les sessions',
      description: (
        <>
          Déconnecter tous les appareils de « {account.username || account.email} ».
          Le mot de passe reste inchangé.
        </>
      ),
      confirmationLabel: 'Révoquer',
      reason: { required: true, label: 'Motif', minLength: 5 },
      onConfirm: async ({ reason }) => {
        await revokeUserSessions(accountId, { reason });
        setActionDialog(null);
        await reload();
        onChanged?.();
      },
    });
  };

  const handleDriverStatus = (nextActive) => {
    if (!driverProfile) return;
    setActionDialog({
      kind: 'driver-status',
      title: nextActive ? 'Réactiver le chauffeur' : 'Désactiver le chauffeur',
      description: nextActive
        ? 'Le profil chauffeur sera réactivé. La disponibilité opérationnelle reste gérée par l’entreprise.'
        : 'Le chauffeur ne pourra plus se connecter. Les sessions seront révoquées. L’historique est conservé.',
      confirmationLabel: nextActive ? 'Réactiver' : 'Désactiver',
      reason: { required: true, label: 'Motif', minLength: 5 },
      onConfirm: async ({ reason }) => {
        await updateDriverStatus(accountId, {
          is_active: nextActive,
          expected_is_active: Boolean(driverProfile.is_active),
          reason,
        });
        setActionDialog(null);
        await reload();
        onChanged?.();
      },
    });
  };

  const headerTitle = showCompanySupport
    ? companyProfile.name || account.username || account.email
    : account.username || account.email || `Compte #${accountId}`;

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
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="manage-account-title"
      >
        <header className={styles.header}>
          <div className={styles.headerText}>
            <p className={styles.eyebrow}>
              {showCompanySupport ? 'Entreprise de transport' : 'Compte'}
            </p>
            <h2 id="manage-account-title" className={styles.title}>
              {headerTitle}
            </h2>
            {showCompanySupport ? (
              <p className={styles.subtitle}>
                Compte propriétaire : {account.email || '—'}
              </p>
            ) : null}
            <div className={styles.metaRow}>
              {!showCompanySupport ? (
                <span className={`${styles.badge} ${styles.badgeRole}`}>
                  {account.role || '—'}
                </span>
              ) : null}
              {account.account_status && !showCompanySupport ? (
                <span className={`${styles.badge} ${styles.badgeNeutral}`}>
                  {account.account_status}
                </span>
              ) : null}
              {passwordTemporary ? (
                <span className={`${styles.badge} ${styles.badgePartial}`}>
                  Mot de passe temporaire
                </span>
              ) : null}
              {driverProfile ? (
                <span
                  className={`${styles.badge} ${
                    driverProfile.is_active ? styles.badgeActive : styles.badgeFull
                  }`}
                >
                  {driverProfile.is_active ? 'Profil actif' : 'Profil inactif'}
                </span>
              ) : null}
              {showCompanySupport ? (
                <>
                  <span
                    className={`${styles.badge} ${
                      companyProfile.is_approved
                        ? styles.badgeActive
                        : styles.badgeNeutral
                    }`}
                  >
                    {companyProfile.is_approved ? 'Approuvée' : 'Non approuvée'}
                  </span>
                  <span
                    className={`${styles.badge} ${
                      companyProfile.dispatch_enabled
                        ? styles.badgeActive
                        : styles.badgeNeutral
                    }`}
                  >
                    {companyProfile.dispatch_enabled
                      ? 'Dispatch activé'
                      : 'Dispatch désactivé'}
                  </span>
                  <span
                    className={`${styles.badge} ${
                      companyProfile.platform_suspended
                        ? styles.badgeFull
                        : styles.badgeActive
                    }`}
                  >
                    {companyProfile.platform_suspended
                      ? 'Plateforme suspendue'
                      : 'Plateforme active'}
                  </span>
                  <span
                    className={`${styles.badge} ${restrictionBadgeClass(restrictionState)}`}
                  >
                    {formatRestrictionLabel(restrictionState)}
                  </span>
                </>
              ) : null}
            </div>
            {driverProfile?.company_name ? (
              <p className={styles.subtitle}>{driverProfile.company_name}</p>
            ) : null}
          </div>
          <button
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
            aria-label="Fermer"
          >
            <FiX size={18} aria-hidden />
          </button>
        </header>

        <div className={styles.body}>
          {loading ? <p className={styles.loading}>Chargement…</p> : null}
          {error ? (
            <p className={styles.error} role="alert">
              {error}
            </p>
          ) : null}

          {ctx ? (
            <>
              <section className={styles.card}>
                <h3 className={styles.cardTitle}>Identité</h3>
                <dl className={styles.kv}>
                  <div>
                    <dt>E-mail</dt>
                    <dd>{account.email || '—'}</dd>
                  </div>
                  <div>
                    <dt>Créé le</dt>
                    <dd>{formatDate(account.created_at)}</dd>
                  </div>
                  {showCompanySupport ? (
                    <div>
                      <dt>Company ID</dt>
                      <dd>{companyProfile.company_id}</dd>
                    </div>
                  ) : null}
                  <div>
                    <dt>Sessions actives</dt>
                    <dd>{ctx.security?.active_sessions ?? '—'}</dd>
                  </div>
                </dl>
              </section>

              {driverProfile ? (
                <section className={styles.card}>
                  <h3 className={styles.cardTitle}>Profil chauffeur</h3>
                  <dl className={styles.kv}>
                    <div>
                      <dt>Entreprise</dt>
                      <dd>{driverProfile.company_name || '—'}</dd>
                    </div>
                    <div>
                      <dt>État profil</dt>
                      <dd>{driverProfile.is_active ? 'Actif' : 'Inactif'}</dd>
                    </div>
                    <div>
                      <dt>Disponibilité</dt>
                      <dd>
                        {driverProfile.is_available ? 'Disponible' : 'Indisponible'}
                      </dd>
                    </div>
                    <div>
                      <dt>Type</dt>
                      <dd>{driverProfile.driver_type || '—'}</dd>
                    </div>
                  </dl>
                  <p className={styles.sectionLead}>
                    La disponibilité opérationnelle est gérée par l&apos;entreprise et
                    l&apos;application chauffeur.
                  </p>
                  {canDriverStatus ? (
                    <div className={styles.actions}>
                      {driverProfile.is_active ? (
                        <button
                          type="button"
                          className={styles.dangerButton}
                          onClick={() => handleDriverStatus(false)}
                        >
                          Désactiver le chauffeur
                        </button>
                      ) : (
                        <button
                          type="button"
                          className={styles.primaryButton}
                          onClick={() => handleDriverStatus(true)}
                        >
                          Réactiver le chauffeur
                        </button>
                      )}
                    </div>
                  ) : null}
                </section>
              ) : null}

              {showCommercial ? (
                <section className={styles.card}>
                  <h3 className={styles.cardTitle}>Restriction commerciale LIRIE</h3>
                  <p className={styles.sectionLead}>
                    Cette restriction concerne le recouvrement LIRIE. Elle ne désactive
                    pas le compte, les chauffeurs, le GPS, les factures ni les courses
                    déjà créées.
                  </p>
                  <dl className={styles.kv}>
                    <div>
                      <dt>État actuel</dt>
                      <dd>
                        <span
                          className={`${styles.badge} ${restrictionBadgeClass(
                            restrictionState
                          )}`}
                        >
                          {formatRestrictionLabel(restrictionState)}
                        </span>
                      </dd>
                    </div>
                  </dl>
                  <p className={styles.muted}>
                    {RESTRICTION_IMPACT[restrictionState] || RESTRICTION_IMPACT.active}
                  </p>
                  {canCommercial ? (
                    <div
                      className={styles.segmentedControl}
                      role="group"
                      aria-label="Niveau de restriction commerciale"
                    >
                      {[
                        { key: 'active', label: 'Aucune restriction' },
                        { key: 'partial', label: 'Partielle' },
                        { key: 'full', label: 'Complète' },
                      ].map((opt) => (
                        <button
                          key={opt.key}
                          type="button"
                          className={
                            restrictionState === opt.key
                              ? styles.segmentActive
                              : styles.segment
                          }
                          aria-pressed={restrictionState === opt.key}
                          onClick={() => {
                            if (restrictionState !== opt.key) {
                              handleCommercialRestriction(opt.key);
                            }
                          }}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className={styles.muted}>
                      Capacité <code>admin.billing.lock</code> requise.
                    </p>
                  )}

                  <div className={styles.subBlock}>
                    <h4 className={styles.subTitle}>Recouvrement automatique</h4>
                    <p className={styles.muted}>
                      {dunningPaused ? (
                        <>
                          En pause jusqu&apos;au {formatDate(dunningPausedUntil)}
                          {commercialRestriction?.dunning_pause_reason
                            ? ` — ${commercialRestriction.dunning_pause_reason}`
                            : ''}
                        </>
                      ) : (
                        <>
                          Actif
                          <br />
                          Aucune pause programmée
                        </>
                      )}
                    </p>
                    {canDunning ? (
                      <div className={styles.actions}>
                        <button
                          type="button"
                          className={styles.secondaryButton}
                          onClick={() => handlePauseDunning({ resume: false })}
                        >
                          {dunningPaused
                            ? 'Modifier la pause'
                            : 'Mettre en pause le recouvrement'}
                        </button>
                        {dunningPaused ? (
                          <button
                            type="button"
                            className={styles.ghostButton}
                            onClick={() => handlePauseDunning({ resume: true })}
                          >
                            Reprendre maintenant
                          </button>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                </section>
              ) : null}

              {showCompanySupport ? (
                <section className={styles.card}>
                  <h3 className={styles.cardTitle}>Gouvernance opérationnelle</h3>
                  <dl className={styles.kv}>
                    <div>
                      <dt>Approbation plateforme</dt>
                      <dd>
                        {companyProfile.is_approved ? 'Approuvée' : 'Non approuvée'}
                      </dd>
                    </div>
                    <div>
                      <dt>Dispatch</dt>
                      <dd>
                        {companyProfile.dispatch_enabled ? 'Activé' : 'Désactivé'}
                      </dd>
                    </div>
                    <div>
                      <dt>Suspension plateforme</dt>
                      <dd>
                        {companyProfile.platform_suspended
                          ? 'Suspendue'
                          : 'Non suspendue'}
                      </dd>
                    </div>
                  </dl>
                  <ul className={styles.helpList}>
                    <li>
                      <strong>Approbation</strong> : entreprise validée par LIRIE.
                    </li>
                    <li>
                      <strong>Dispatch</strong> : accès aux fonctions d&apos;exploitation
                      et d&apos;affectation.
                    </li>
                    <li>
                      <strong>Suspension plateforme</strong> : blocage global de
                      gouvernance, y compris la connexion.
                    </li>
                  </ul>
                  {canOpsFlags ? (
                    <div className={styles.actions}>
                      <button
                        type="button"
                        className={styles.secondaryButton}
                        onClick={() =>
                          handleApproval(!Boolean(companyProfile.is_approved))
                        }
                      >
                        {companyProfile.is_approved
                          ? 'Retirer l’approbation'
                          : 'Approuver'}
                      </button>
                      <button
                        type="button"
                        className={styles.secondaryButton}
                        onClick={() =>
                          handleDispatch(!Boolean(companyProfile.dispatch_enabled))
                        }
                      >
                        {companyProfile.dispatch_enabled
                          ? 'Désactiver le dispatch'
                          : 'Activer le dispatch'}
                      </button>
                    </div>
                  ) : (
                    <p className={styles.muted}>
                      Modification des drapeaux : capacité{' '}
                      <code>admin.users.manage</code> requise.
                    </p>
                  )}
                  {platformOpsPath ? (
                    <Link
                      to={platformOpsPath}
                      className={styles.textLink}
                      onClick={onClose}
                    >
                      Ouvrir dans Platform Ops
                    </Link>
                  ) : null}
                </section>
              ) : null}

              {showCompanySupport ? (
                <section className={styles.card}>
                  <h3 className={styles.cardTitle}>Chauffeurs</h3>
                  <p className={styles.sectionLead}>
                    Compteurs informatifs uniquement — aucun quota ni licence.
                  </p>
                  <dl className={styles.kv}>
                    <div>
                      <dt>Actifs</dt>
                      <dd>{companyProfile.active_drivers_count ?? 0}</dd>
                    </div>
                    <div>
                      <dt>Inactifs</dt>
                      <dd>{companyProfile.inactive_drivers_count ?? 0}</dd>
                    </div>
                    <div>
                      <dt>Total</dt>
                      <dd>{companyProfile.total_drivers_count ?? 0}</dd>
                    </div>
                  </dl>
                  {driversListPath ? (
                    <Link
                      to={driversListPath}
                      className={styles.textLink}
                      onClick={onClose}
                    >
                      Voir les comptes chauffeurs
                    </Link>
                  ) : null}
                </section>
              ) : null}

              {showCompanySupport && detectedServices ? (
                <section className={styles.card}>
                  <h3 className={styles.cardTitle}>Services détectés</h3>
                  <dl className={styles.kv}>
                    {(detectedServices.services || []).map((svc) => (
                      <div key={svc.service_key}>
                        <dt>{svc.label}</dt>
                        <dd>
                          <span className={`${styles.badge} ${styles.badgeInfo}`}>
                            Détecté
                          </span>
                        </dd>
                      </div>
                    ))}
                  </dl>
                  <p className={styles.muted}>
                    Mode : Shadow — non appliqué
                  </p>
                  <p className={styles.noticeInfo}>
                    {detectedServices.notice ||
                      'Ces services sont détectés depuis la configuration legacy. Ils n’autorisent ni ne bloquent encore les fonctions de l’entreprise.'}
                  </p>
                </section>
              ) : null}

              {showCompanySupport && (financeConfigPath || platformOpsPath) ? (
                <section className={styles.card}>
                  <h3 className={styles.cardTitle}>Raccourcis</h3>
                  <div className={styles.linkStack}>
                    {financeConfigPath ? (
                      <Link
                        to={financeConfigPath}
                        className={styles.textLink}
                        onClick={onClose}
                        state={{ companyId: companyProfile.company_id }}
                      >
                        Ouvrir la configuration Billing
                      </Link>
                    ) : null}
                    {platformOpsPath ? (
                      <Link
                        to={platformOpsPath}
                        className={styles.textLink}
                        onClick={onClose}
                      >
                        Ouvrir dans Platform Ops
                      </Link>
                    ) : null}
                  </div>
                </section>
              ) : null}

              <section className={styles.card}>
                <h3 className={styles.cardTitle}>Sécurité du compte</h3>
                <div className={styles.actions}>
                  {canReset ? (
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      onClick={handleResetPassword}
                    >
                      Réinitialiser le mot de passe
                    </button>
                  ) : null}
                  {canRevoke ? (
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      onClick={handleRevokeSessions}
                    >
                      Révoquer les sessions
                    </button>
                  ) : null}
                </div>
                {!canReset && !canRevoke ? (
                  <p className={styles.muted}>
                    Capacité <code>admin.users.security</code> requise.
                  </p>
                ) : null}
              </section>

              <section className={styles.card}>
                <h3 className={styles.cardTitle}>Rôle et rattachement</h3>
                {canRole ? (
                  <>
                    <div className={styles.field}>
                      <label className={styles.fieldLabel} htmlFor={`${reasonId}-role`}>
                        Nouveau rôle
                      </label>
                      <select
                        id={`${reasonId}-role`}
                        value={roleDraft}
                        onChange={(e) => setRoleDraft(e.target.value)}
                      >
                        {ROLE_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {roleDraft === 'driver' || roleDraft === 'company' ? (
                      <div className={styles.field}>
                        <label className={styles.fieldLabel} htmlFor={`${reasonId}-co`}>
                          Entreprise (tenant transport)
                        </label>
                        <select
                          id={`${reasonId}-co`}
                          value={companyId}
                          onChange={(e) => setCompanyId(e.target.value)}
                        >
                          <option value="">— Sélectionner —</option>
                          {(options.transport_tenants || []).map((c) => (
                            <option key={c.id} value={c.id}>
                              {c.name}
                            </option>
                          ))}
                        </select>
                      </div>
                    ) : null}

                    {roleDraft === 'institution' ? (
                      <>
                        <div className={styles.field}>
                          <label
                            className={styles.fieldLabel}
                            htmlFor={`${reasonId}-inst`}
                          >
                            Institution
                          </label>
                          <select
                            id={`${reasonId}-inst`}
                            value={institutionId}
                            onChange={(e) => setInstitutionId(e.target.value)}
                          >
                            <option value="">— Sélectionner —</option>
                            {(options.institutions || []).map((i) => (
                              <option key={i.id} value={i.id}>
                                {i.name}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className={styles.field}>
                          <label
                            className={styles.fieldLabel}
                            htmlFor={`${reasonId}-irole`}
                          >
                            Rôle institutionnel
                          </label>
                          <select
                            id={`${reasonId}-irole`}
                            value={institutionRole}
                            onChange={(e) => setInstitutionRole(e.target.value)}
                          >
                            {(options.institution_roles || []).map((r) => (
                              <option key={r} value={r}>
                                {INSTITUTION_ROLE_LABELS[r] || r}
                              </option>
                            ))}
                          </select>
                        </div>
                      </>
                    ) : null}

                    <button
                      type="button"
                      className={styles.primaryButton}
                      disabled={busy}
                      onClick={handleApplyRole}
                    >
                      {busy ? 'Prévisualisation…' : 'Prévisualiser et appliquer'}
                    </button>
                    {isCompanyRole ? (
                      <p className={styles.notice}>
                        Quitter le rôle Entreprise tant que le compte est propriétaire
                        d&apos;un tenant est bloqué (assistant ownership CP-PR3).
                      </p>
                    ) : null}
                  </>
                ) : (
                  <p className={styles.muted}>
                    Rôle actuel : {account.role}. Capacité{' '}
                    <code>admin.users.manage</code> requise pour modifier.
                  </p>
                )}
              </section>

              <section className={styles.card}>
                <h3 className={styles.cardTitle}>Diagnostic</h3>
                {(diagnostic.checks || []).length ? (
                  <ul className={styles.checks}>
                    {(diagnostic.checks || []).map((check) => (
                      <li key={check.code} className={checkClass(check.status)}>
                        {check.label}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className={styles.muted}>Aucun contrôle disponible.</p>
                )}
              </section>
            </>
          ) : null}
        </div>

        <footer className={styles.footer}>
          <button type="button" className={styles.ghostButton} onClick={onClose}>
            Fermer
          </button>
        </footer>
      </aside>

      {actionDialog ? (
        <AdminActionDialog
          open
          title={actionDialog.title}
          description={actionDialog.description}
          impact={actionDialog.impact}
          confirmationLabel={actionDialog.confirmationLabel}
          confirmText={actionDialog.confirmText}
          reason={actionDialog.reason}
          onConfirm={actionDialog.onConfirm}
          onClose={() => setActionDialog(null)}
        />
      ) : null}

      <AdminTempPasswordDialog
        open={Boolean(tempPasswordDialog)}
        accountLabel={tempPasswordDialog?.accountLabel || ''}
        temporaryPassword={tempPasswordDialog?.temporaryPassword || ''}
        onClose={() => setTempPasswordDialog(null)}
      />
    </div>
  );

  return createPortal(drawer, document.body);
};

export default AdminAccountManageDrawer;
