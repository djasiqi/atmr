// pages/institution/Settings/components/UsersRolesTab.jsx
/**
 * Onglet "Utilisateurs & accès" pour les paramètres institution.
 *
 * P2: Invitation par email
 * - Inviter un utilisateur → envoi email avec lien d'activation
 * - Colonne "Statut" (Invité / Actif / Désactivé)
 * - Bouton "Renvoyer invitation" pour les invités
 * - Bouton "Désactiver" au lieu de retirer
 * - Règle "dernier admin": impossible de rétrograder/désactiver le dernier admin
 */

import React, { useState } from 'react';
import { FaCopy, FaTrash, FaTimes, FaEnvelope, FaBan, FaRedo, FaLink, FaCheckCircle, FaTimesCircle, FaShieldAlt } from 'react-icons/fa';
import {
  useInstitutionUsers,
  useInviteInstitutionUser,
  useUpdateUserRole,
  useRemoveInstitutionUser,
  useResendInvite,
  useDisableInstitutionUser,
  useInstitutionMe,
  usePermissionRequests,
  useResolvePermissionRequest,
} from '../../../../hooks/useInstitutionData';
import { isAdmin } from '../../../../utils/institutionPermissions';
import { toast } from 'sonner';
import styles from '../InstitutionSettings.module.css';

/** Copie un texte dans le presse-papier et affiche un toast. */
const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text);
    toast.success('Lien copié dans le presse-papier');
  } catch {
    // Fallback pour navigateurs sans API clipboard
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    toast.success('Lien copié');
  }
};

const BASE_ROLE_OPTIONS = [
  { value: 'institution_admin', label: 'Administrateur', labelCuratelle: 'Direction', desc: 'Accès complet : paramètres, utilisateurs, transporteurs, clés API', descCuratelle: 'Accès complet : gestion des équipes, curateurs, paramètres et supervision' },
  { value: 'institution_curator', label: 'Curateur', desc: 'Gère les demandes et la facturation pour ses protégés (scope équipe)', curatelle_only: true },
  { value: 'institution_requester', label: 'Demandeur', desc: 'Créer et envoyer des demandes de transport, gérer les patients' },
  { value: 'institution_billing', label: 'Facturation', desc: 'Gérer la facturation et les paramètres de billing' },
  { value: 'institution_reader', label: 'Lecteur', desc: 'Consultation uniquement, aucune modification' },
];

const getRoleOptions = (institutionType) => {
  const isCuratelle = institutionType?.toLowerCase() === 'curatelle';
  return BASE_ROLE_OPTIONS
    .filter((r) => isCuratelle || !r.curatelle_only)
    .map((r) => ({
      value: r.value,
      label: isCuratelle && r.labelCuratelle ? r.labelCuratelle : r.label,
      desc: isCuratelle && r.descCuratelle ? r.descCuratelle : r.desc,
    }));
};

const STATUS_CONFIG = {
  invited: { label: 'Invité', bg: '#fff3e0', color: '#e65100', icon: '📧' },
  expired: { label: 'Expiré', bg: '#fce4ec', color: '#ad1457', icon: '⏱' },
  active: { label: 'Actif', bg: '#e8f5e9', color: '#2e7d32', icon: '✓' },
  disabled: { label: 'Désactivé', bg: '#ffebee', color: '#c62828', icon: '✗' },
};

const getRoleBadgeStyle = (role) => {
  const colors = {
    institution_admin: { bg: '#e3f2fd', color: '#1565c0' },
    institution_curator: { bg: '#ede7f6', color: '#7C3AED' },
    institution_requester: { bg: '#e8f5e9', color: '#2e7d32' },
    institution_billing: { bg: '#fff3e0', color: '#e65100' },
    institution_reader: { bg: '#f5f5f5', color: '#616161' },
  };
  return colors[role] || colors.institution_reader;
};

const getRoleLabelStatic = (role) => {
  const opt = BASE_ROLE_OPTIONS.find((r) => r.value === role);
  return opt ? opt.label : role || 'Inconnu';
};

const getStatusConfig = (status) => {
  return STATUS_CONFIG[status] || STATUS_CONFIG.active;
};

const UsersRolesTab = () => {
  const { data: meData } = useInstitutionMe();
  const { data: usersData, isLoading } = useInstitutionUsers();
  const { data: permRequestsData } = usePermissionRequests();
  const inviteMutation = useInviteInstitutionUser();
  const updateRoleMutation = useUpdateUserRole();
  const removeMutation = useRemoveInstitutionUser();
  const resendMutation = useResendInvite();
  const disableMutation = useDisableInstitutionUser();
  const resolvePermMutation = useResolvePermissionRequest();

  const canEdit = isAdmin(meData?.institution_role);
  const currentUserId = meData?.user?.id;
  const institutionType = meData?.institution_type;
  const roleOptions = getRoleOptions(institutionType);
  const pendingPermRequests = (permRequestsData?.requests || []).filter(r => r.status === 'pending');

  const getRoleLabel = (role) => {
    const opt = roleOptions.find((r) => r.value === role);
    return opt ? opt.label : getRoleLabelStatic(role);
  };

  // Invite form state
  const [showInvite, setShowInvite] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState('institution_requester');
  const [inviteFirstName, setInviteFirstName] = useState('');
  const [inviteLastName, setInviteLastName] = useState('');

  // Confirmation modal state
  const [confirmAction, setConfirmAction] = useState(null);

  // Résultat de la dernière invitation (pour afficher le lien fallback)
  const [lastInviteResult, setLastInviteResult] = useState(null);

  const users = usersData?.users || [];

  const handleInvite = async (e) => {
    e.preventDefault();
    if (!inviteEmail.trim()) {
      toast.error('L\'email est requis');
      return;
    }

    try {
      const result = await inviteMutation.mutateAsync({
        email: inviteEmail.trim().toLowerCase(),
        institution_role: inviteRole,
        first_name: inviteFirstName.trim() || undefined,
        last_name: inviteLastName.trim() || undefined,
      });

      if (result.email_sent) {
        toast.success('Invitation envoyée par email');
      } else if (result.email_sent === false) {
        toast.warning('Email non envoyé — utilisez le lien ci-dessous.');
      } else {
        toast.success(result.message || 'Utilisateur ajouté');
      }

      // Afficher le résultat avec le lien (toujours, même si email OK)
      if (result.invite_link) {
        setLastInviteResult({
          email: inviteEmail.trim().toLowerCase(),
          emailSent: result.email_sent,
          emailError: result.email_error,
          inviteLink: result.invite_link,
        });
      }

      // Reset form
      setInviteEmail('');
      setInviteRole('institution_requester');
      setInviteFirstName('');
      setInviteLastName('');
      setShowInvite(false);
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors de l\'invitation';
      toast.error(msg);
    }
  };

  const handleRoleChange = async (userId, newRole) => {
    try {
      await updateRoleMutation.mutateAsync({ userId, institution_role: newRole });
      toast.success('Rôle mis à jour');
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors de la mise à jour du rôle';
      toast.error(msg);
    }
  };

  const handleRemove = async (userId) => {
    try {
      await removeMutation.mutateAsync(userId);
      toast.success('Utilisateur retiré de l\'institution');
      setConfirmAction(null);
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors du retrait';
      toast.error(msg);
      setConfirmAction(null);
    }
  };

  const handleResendInvite = async (userId) => {
    try {
      const result = await resendMutation.mutateAsync(userId);

      if (result.email_sent) {
        toast.success('Invitation renvoyée par email');
      } else {
        toast.warning('Email non envoyé — utilisez le lien ci-dessous.');
      }

      // Afficher le lien fallback
      if (result.invite_link) {
        setLastInviteResult({
          email: result.user?.email || '',
          emailSent: result.email_sent,
          emailError: result.email_error,
          inviteLink: result.invite_link,
        });
      }
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors du renvoi de l\'invitation';
      toast.error(msg);
    }
  };

  const handleDisable = async (userId) => {
    try {
      await disableMutation.mutateAsync(userId);
      toast.success('Utilisateur désactivé');
      setConfirmAction(null);
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors de la désactivation';
      toast.error(msg);
      setConfirmAction(null);
    }
  };

  if (isLoading) {
    return (
      <div className={styles.section}>
        <p>Chargement des utilisateurs...</p>
      </div>
    );
  }

  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3>Utilisateurs & accès</h3>
        <p>Gérez les membres de votre institution et leurs permissions.</p>
      </div>

      {/* Légende des rôles */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 10,
        marginBottom: 16,
        padding: '10px 14px',
        background: '#f8f9fa',
        borderRadius: 8,
        border: '1px solid #eee',
        fontSize: 12,
        color: '#555',
      }}>
        {roleOptions.map((r) => {
          const badge = getRoleBadgeStyle(r.value);
          return (
            <span key={r.value} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <span style={{
                display: 'inline-block',
                padding: '2px 8px',
                borderRadius: 10,
                fontSize: 11,
                fontWeight: 600,
                backgroundColor: badge.bg,
                color: badge.color,
              }}>
                {r.label}
              </span>
              <span style={{ color: '#888' }}>{r.desc}</span>
            </span>
          );
        })}
      </div>

      {/* Action bar */}
      {canEdit && (
        <div style={{ marginBottom: 16 }}>
          <button
            className={styles.saveBtn}
            onClick={() => setShowInvite(!showInvite)}
          >
            {showInvite ? <FaTimes /> : <FaEnvelope />}
            {showInvite ? ' Annuler' : ' Inviter par email'}
          </button>
        </div>
      )}

      {/* Invite form */}
      {showInvite && canEdit && (
        <form onSubmit={handleInvite} style={{
          background: '#fafafa',
          padding: 20,
          borderRadius: 8,
          marginBottom: 20,
          border: '1px solid #e0e0e0',
        }}>
          <div style={{ marginBottom: 12, fontSize: 13, color: '#666' }}>
            <FaEnvelope style={{ marginRight: 6, verticalAlign: 'middle' }} />
            Un email d'invitation sera envoyé avec un lien d'activation (valable 48h).
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
            <div>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Email *
              </label>
              <input
                type="email"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                placeholder="email@exemple.ch"
                required
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #ddd',
                  borderRadius: 6,
                  fontSize: 14,
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Rôle *
              </label>
              <select
                value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #ddd',
                  borderRadius: 6,
                  fontSize: 14,
                  boxSizing: 'border-box',
                }}
              >
                {roleOptions.map((r) => (
                  <option key={r.value} value={r.value} title={r.desc}>{r.label}</option>
                ))}
              </select>
              <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>
                {roleOptions.find((r) => r.value === inviteRole)?.desc || ''}
              </div>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
            <div>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Prénom
              </label>
              <input
                type="text"
                value={inviteFirstName}
                onChange={(e) => setInviteFirstName(e.target.value)}
                placeholder="Prénom"
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #ddd',
                  borderRadius: 6,
                  fontSize: 14,
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Nom
              </label>
              <input
                type="text"
                value={inviteLastName}
                onChange={(e) => setInviteLastName(e.target.value)}
                placeholder="Nom"
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #ddd',
                  borderRadius: 6,
                  fontSize: 14,
                  boxSizing: 'border-box',
                }}
              />
            </div>
          </div>
          <button
            type="submit"
            className={styles.saveBtn}
            disabled={inviteMutation.isPending}
          >
            <FaEnvelope />
            {inviteMutation.isPending ? ' Envoi en cours...' : ' Envoyer l\'invitation'}
          </button>
        </form>
      )}

      {/* Invite result banner (lien fallback) */}
      {lastInviteResult && (
        <div style={{
          background: lastInviteResult.emailSent ? '#e8f5e9' : '#fff8e1',
          border: `1px solid ${lastInviteResult.emailSent ? '#c8e6c9' : '#ffe082'}`,
          borderRadius: 8,
          padding: '14px 18px',
          marginBottom: 16,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>
                {lastInviteResult.emailSent
                  ? `✅ Invitation envoyée à ${lastInviteResult.email}`
                  : `⚠️ Email non envoyé à ${lastInviteResult.email}`}
              </div>
              {!lastInviteResult.emailSent && lastInviteResult.emailError && (
                <div style={{ fontSize: 12, color: '#b71c1c', marginBottom: 6 }}>
                  Raison : {lastInviteResult.emailError}
                </div>
              )}
              <div style={{ fontSize: 13, color: '#555', marginBottom: 8 }}>
                {lastInviteResult.emailSent
                  ? 'Vous pouvez aussi partager ce lien manuellement si besoin :'
                  : 'Transmettez ce lien manuellement à l\'utilisateur :'}
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                background: '#fff',
                border: '1px solid #ddd',
                borderRadius: 6,
                padding: '6px 10px',
                maxWidth: 500,
              }}>
                <FaLink style={{ color: '#667eea', flexShrink: 0 }} />
                <code style={{
                  fontSize: 12,
                  color: '#333',
                  wordBreak: 'break-all',
                  flex: 1,
                }}>
                  {lastInviteResult.inviteLink}
                </code>
                <button
                  onClick={() => copyToClipboard(lastInviteResult.inviteLink)}
                  title="Copier le lien"
                  style={{
                    padding: '4px 10px',
                    background: '#667eea',
                    color: '#fff',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer',
                    fontSize: 12,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    flexShrink: 0,
                  }}
                >
                  <FaCopy /> Copier
                </button>
              </div>
            </div>
            <button
              onClick={() => setLastInviteResult(null)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#999', padding: 4 }}
              title="Fermer"
            >
              <FaTimes />
            </button>
          </div>
        </div>
      )}

      {/* Demandes de droits en attente (admin only) */}
      {canEdit && pendingPermRequests.length > 0 && (
        <div style={{
          marginBottom: 20,
          padding: 16,
          background: '#fff8e1',
          border: '1px solid #ffe082',
          borderRadius: 8,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <FaShieldAlt style={{ color: '#e65100' }} />
            <h4 style={{ margin: 0, fontSize: 14, color: '#5d4037' }}>
              Demandes de droits en attente ({pendingPermRequests.length})
            </h4>
          </div>
          {pendingPermRequests.map((req) => (
            <div key={req.id} style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '10px 14px',
              background: '#fff',
              borderRadius: 6,
              marginBottom: 6,
              border: '1px solid #ffe082',
            }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: '#333' }}>
                  {req.user_name || req.user_email}
                  <span style={{ fontWeight: 400, color: '#888', marginLeft: 6, fontSize: 12 }}>
                    ({getRoleLabel(req.current_role)} → {getRoleLabel(req.requested_role)})
                  </span>
                </div>
                <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>
                  « {req.message} »
                </div>
                <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
                  {new Date(req.created_at).toLocaleDateString('fr-CH', {
                    day: '2-digit', month: '2-digit', year: 'numeric',
                    hour: '2-digit', minute: '2-digit',
                  })}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  onClick={() => resolvePermMutation.mutate({ requestId: req.id, action: 'approve' })}
                  disabled={resolvePermMutation.isPending}
                  title="Approuver"
                  style={{
                    padding: '6px 12px',
                    background: '#e8f5e9',
                    border: '1px solid #4caf50',
                    borderRadius: 6,
                    color: '#2e7d32',
                    cursor: 'pointer',
                    fontSize: 12,
                    fontWeight: 500,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                  }}
                >
                  <FaCheckCircle size={12} /> Approuver
                </button>
                <button
                  onClick={() => resolvePermMutation.mutate({ requestId: req.id, action: 'deny' })}
                  disabled={resolvePermMutation.isPending}
                  title="Refuser"
                  style={{
                    padding: '6px 12px',
                    background: '#ffebee',
                    border: '1px solid #ef5350',
                    borderRadius: 6,
                    color: '#c62828',
                    cursor: 'pointer',
                    fontSize: 12,
                    fontWeight: 500,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                  }}
                >
                  <FaTimesCircle size={12} /> Refuser
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Users table */}
      {users.length === 0 ? (
        <p className={styles.emptyState}>Aucun utilisateur dans cette institution.</p>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #eee' }}>
              <th style={thStyle}>Utilisateur</th>
              <th style={thStyle}>Rôle</th>
              <th style={thStyle}>Statut</th>
              <th style={thStyle}>Ajouté le</th>
              {canEdit && <th style={{ ...thStyle, textAlign: 'right' }}>Actions</th>}
            </tr>
          </thead>
          <tbody>
            {users.map((user) => {
              const isSelf = user.id === currentUserId;
              const badgeStyle = getRoleBadgeStyle(user.institution_role);
              const status = getStatusConfig(user.account_status);
              const isInvited = user.account_status === 'invited' || user.account_status === 'expired';
              const isDisabled = user.account_status === 'disabled';

              return (
                <tr
                  key={user.id}
                  style={{
                    borderBottom: '1px solid #f0f0f0',
                    opacity: isDisabled ? 0.6 : 1,
                  }}
                >
                  <td style={{ padding: '12px' }}>
                    <div style={{ fontWeight: 500, fontSize: 14, color: '#333' }}>
                      {user.first_name || user.last_name
                        ? `${user.first_name || ''} ${user.last_name || ''}`.trim()
                        : user.username || '-'}
                      {isSelf && (
                        <span style={{ marginLeft: 8, fontSize: 11, color: '#999', fontWeight: 400 }}>
                          (vous)
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 13, color: '#888' }}>{user.email}</div>
                  </td>
                  <td style={{ padding: '12px' }}>
                    {canEdit && !isSelf && !isDisabled ? (
                      <select
                        value={user.institution_role || ''}
                        onChange={(e) => handleRoleChange(user.id, e.target.value)}
                        disabled={updateRoleMutation.isPending}
                        title={roleOptions.find((r) => r.value === user.institution_role)?.desc || ''}
                        style={{
                          padding: '6px 10px',
                          border: '1px solid #ddd',
                          borderRadius: 6,
                          fontSize: 13,
                          background: 'white',
                        }}
                      >
                        {roleOptions.map((r) => (
                          <option key={r.value} value={r.value} title={r.desc}>{r.label}</option>
                        ))}
                      </select>
                    ) : (
                      <span
                        title={roleOptions.find((r) => r.value === user.institution_role)?.desc || ''}
                        style={{
                          display: 'inline-block',
                          padding: '4px 12px',
                          borderRadius: 12,
                          fontSize: 12,
                          fontWeight: 500,
                          backgroundColor: badgeStyle.bg,
                          color: badgeStyle.color,
                      }}>
                        {getRoleLabel(user.institution_role)}
                      </span>
                    )}
                  </td>
                  <td style={{ padding: '12px' }}>
                    <span style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '4px 10px',
                      borderRadius: 12,
                      fontSize: 12,
                      fontWeight: 500,
                      backgroundColor: status.bg,
                      color: status.color,
                    }}>
                      <span>{status.icon}</span>
                      {status.label}
                    </span>
                    {isInvited && user.invite_sent_at && (
                      <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
                        Envoyé {new Date(user.invite_sent_at).toLocaleDateString('fr-CH')}
                      </div>
                    )}
                  </td>
                  <td style={{ padding: '12px', fontSize: 13, color: '#666' }}>
                    {user.created_at
                      ? new Date(user.created_at).toLocaleDateString('fr-CH')
                      : '-'}
                  </td>
                  {canEdit && (
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                        {/* Renvoyer invitation - seulement pour invited/disabled */}
                        {(isInvited || isDisabled) && (
                          <button
                            onClick={() => handleResendInvite(user.id)}
                            disabled={resendMutation.isPending}
                            title={isDisabled ? 'Réactiver et renvoyer invitation' : 'Renvoyer l\'invitation'}
                            style={actionBtnStyle('#1565c0', '#e3f2fd')}
                          >
                            <FaRedo style={{ fontSize: 11 }} />
                          </button>
                        )}
                        {/* Désactiver - pas soi-même, pas déjà disabled */}
                        {!isSelf && !isDisabled && (
                          <button
                            onClick={() => setConfirmAction({ type: 'disable', user })}
                            disabled={disableMutation.isPending}
                            title="Désactiver l'utilisateur"
                            style={actionBtnStyle('#e65100', '#fff3e0')}
                          >
                            <FaBan style={{ fontSize: 11 }} />
                          </button>
                        )}
                        {/* Retirer - pas soi-même */}
                        {!isSelf && (
                          <button
                            onClick={() => setConfirmAction({ type: 'remove', user })}
                            disabled={removeMutation.isPending}
                            title="Retirer de l'institution"
                            style={actionBtnStyle('#c62828', '#ffebee')}
                          >
                            <FaTrash style={{ fontSize: 11 }} />
                          </button>
                        )}
                      </div>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {/* Confirmation modal */}
      {confirmAction && (
        <div className={styles.modal}>
          <div className={styles.modalContent} style={{ maxWidth: 420 }}>
            <div className={styles.modalHeader}>
              <h3>
                {confirmAction.type === 'disable'
                  ? 'Confirmer la désactivation'
                  : 'Confirmer le retrait'}
              </h3>
              <button onClick={() => setConfirmAction(null)}>
                <FaTimes />
              </button>
            </div>
            <div className={styles.modalBody}>
              {confirmAction.type === 'disable' ? (
                <>
                  <p>
                    Voulez-vous vraiment désactiver{' '}
                    <strong>{confirmAction.user.email}</strong> ?
                  </p>
                  <p style={{ fontSize: 13, color: '#666', marginTop: 8 }}>
                    L'utilisateur ne pourra plus se connecter au portail.
                    Vous pourrez réactiver son accès en renvoyant une invitation.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    Voulez-vous vraiment retirer{' '}
                    <strong>{confirmAction.user.email}</strong>{' '}
                    de l'institution ?
                  </p>
                  <p style={{ fontSize: 13, color: '#666', marginTop: 8 }}>
                    L'utilisateur ne pourra plus accéder au portail institution.
                    Son compte ne sera pas supprimé.
                  </p>
                </>
              )}
            </div>
            <div className={styles.modalActions}>
              <button onClick={() => setConfirmAction(null)}>Annuler</button>
              {confirmAction.type === 'disable' ? (
                <button
                  onClick={() => handleDisable(confirmAction.user.id)}
                  style={{ backgroundColor: '#e65100', color: 'white' }}
                  disabled={disableMutation.isPending}
                >
                  {disableMutation.isPending ? 'Désactivation...' : 'Désactiver'}
                </button>
              ) : (
                <button
                  onClick={() => handleRemove(confirmAction.user.id)}
                  style={{ backgroundColor: '#c62828', color: 'white' }}
                  disabled={removeMutation.isPending}
                >
                  {removeMutation.isPending ? 'Retrait...' : 'Retirer'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Style helpers
const thStyle = {
  textAlign: 'left',
  padding: '10px 12px',
  fontSize: 12,
  textTransform: 'uppercase',
  color: '#666',
  fontWeight: 600,
};

const actionBtnStyle = (color, bg) => ({
  padding: '6px 8px',
  background: bg,
  border: `1px solid ${color}33`,
  borderRadius: 6,
  color,
  cursor: 'pointer',
  fontSize: 12,
  display: 'flex',
  alignItems: 'center',
});

export default UsersRolesTab;
