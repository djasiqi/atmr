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
import { FaCopy, FaTrash, FaTimes, FaEnvelope, FaBan, FaRedo, FaLink, FaCheckCircle, FaTimesCircle, FaShieldAlt, FaKey, FaUserPlus, FaEdit } from 'react-icons/fa';
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
  usePendingActivationUsers,
  useResetInstitutionUserPassword,
  useUpdateInstitutionUserProfile,
} from '../../../../hooks/useInstitutionData';
import { isAdmin } from '../../../../utils/institutionPermissions';
import JobTitleCombobox from './JobTitleCombobox';
import ChipSelect from './ChipSelect';
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

const PENDING_REASON_LABELS = {
  never_connected: 'Jamais connecté',
  password_expired: 'Mot de passe expiré',
  invitation_pending: 'Invitation en attente',
  invitation_expired: 'Invitation expirée',
  invited: 'Invité',
};

/** Affiche un toast cohérent avec le contrat API (email_type, creation_mode). */
const showInviteResultToast = (result) => {
  if (result.creation_mode === 'username' && result.temporary_credentials) {
    toast.success('Compte créé — notez les identifiants (affichés une seule fois)');
    return;
  }
  if (result.email_type === 'access_notification') {
    if (result.email_sent) {
      toast.success('Notification d\'accès envoyée — le compte existant reste actif');
    } else {
      toast.warning('Notification non envoyée — informez l\'utilisateur manuellement');
    }
    return;
  }
  if (result.email_type === 'invitation') {
    if (result.email_sent) {
      toast.success('Invitation envoyée par email');
    } else {
      toast.warning('Email non envoyé — utilisez le lien ci-dessous.');
    }
    return;
  }
  if (result.email_sent === false) {
    toast.warning('Email non envoyé');
    return;
  }
  if (result.email_sent === true) {
    toast.success('Email envoyé');
    return;
  }
  toast.success(result.message || 'Utilisateur ajouté');
};

const UsersRolesTab = () => {
  const { data: meData } = useInstitutionMe();
  const { data: usersData, isLoading } = useInstitutionUsers();
  const { data: pendingData } = usePendingActivationUsers();
  const { data: permRequestsData } = usePermissionRequests();
  const inviteMutation = useInviteInstitutionUser();
  const updateRoleMutation = useUpdateUserRole();
  const removeMutation = useRemoveInstitutionUser();
  const resendMutation = useResendInvite();
  const disableMutation = useDisableInstitutionUser();
  const resetPasswordMutation = useResetInstitutionUserPassword();
  const resolvePermMutation = useResolvePermissionRequest();
  const updateProfileMutation = useUpdateInstitutionUserProfile();

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
  const [inviteCreationMode, setInviteCreationMode] = useState('email');
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteUsername, setInviteUsername] = useState('');
  const [inviteRole, setInviteRole] = useState('institution_requester');
  const [inviteFirstName, setInviteFirstName] = useState('');
  const [inviteLastName, setInviteLastName] = useState('');
  const [inviteJobTitle, setInviteJobTitle] = useState('');

  // Confirmation modal state
  const [confirmAction, setConfirmAction] = useState(null);

  // Édition profil utilisateur (admin)
  const [editProfileUser, setEditProfileUser] = useState(null);
  const [editFirstName, setEditFirstName] = useState('');
  const [editLastName, setEditLastName] = useState('');
  const [editEmail, setEditEmail] = useState('');

  // Résultat de la dernière invitation (lien fallback ou credentials one-shot)
  const [lastInviteResult, setLastInviteResult] = useState(null);
  const [credentialsModal, setCredentialsModal] = useState(null);

  const users = usersData?.users || [];
  const pendingUsers = pendingData?.users || [];

  const handleInvite = async (e) => {
    e.preventDefault();

    if (inviteCreationMode === 'email' && !inviteEmail.trim()) {
      toast.error('L\'email est requis');
      return;
    }
    if (inviteCreationMode === 'username' && !inviteUsername.trim()) {
      toast.error('L\'identifiant est requis');
      return;
    }

    try {
      const jobTitle = inviteJobTitle.trim();
      const payload = {
        creation_mode: inviteCreationMode,
        institution_role: inviteRole,
        first_name: inviteFirstName.trim() || undefined,
        last_name: inviteLastName.trim() || undefined,
        job_title: jobTitle || undefined,
      };
      if (inviteCreationMode === 'email') {
        payload.email = inviteEmail.trim().toLowerCase();
      } else {
        payload.username = inviteUsername.trim().toLowerCase();
        if (inviteEmail.trim()) {
          payload.email = inviteEmail.trim().toLowerCase();
        }
      }

      const result = await inviteMutation.mutateAsync(payload);

      showInviteResultToast(result);

      if (result.temporary_credentials) {
        setCredentialsModal({
          ...result.temporary_credentials,
          credentialsShownOnce: result.credentials_shown_once,
        });
      }

      if (result.invite_link) {
        setLastInviteResult({
          email: inviteCreationMode === 'email' ? inviteEmail.trim().toLowerCase() : null,
          emailSent: result.email_sent,
          emailError: result.email_error,
          emailType: result.email_type,
          inviteLink: result.invite_link,
        });
      }

      setInviteEmail('');
      setInviteUsername('');
      setInviteRole('institution_requester');
      setInviteFirstName('');
      setInviteLastName('');
      setInviteJobTitle('');
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

  const openEditProfile = (user) => {
    setEditProfileUser(user);
    setEditFirstName(user.first_name || '');
    setEditLastName(user.last_name || '');
    setEditEmail(user.email || '');
  };

  const handleEditProfileSave = async (e) => {
    e.preventDefault();
    if (!editProfileUser) return;

    try {
      await updateProfileMutation.mutateAsync({
        userId: editProfileUser.id,
        first_name: editFirstName.trim() || null,
        last_name: editLastName.trim() || null,
        email: editEmail.trim() ? editEmail.trim().toLowerCase() : null,
      });
      toast.success('Profil mis à jour');
      setEditProfileUser(null);
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors de la mise à jour du profil';
      toast.error(msg);
    }
  };

  const handleJobTitleSave = async (user, rawValue) => {
    const next = (rawValue || '').replace(/\s+/g, ' ').trim();
    const current = (user.job_title || '').trim();
    if (next === current) return;
    try {
      await updateProfileMutation.mutateAsync({
        userId: user.id,
        job_title: next || null,
      });
      toast.success('Fonction mise à jour');
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors de la mise à jour de la fonction';
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

      if (result.email_type === 'access_notification') {
        if (result.email_sent) {
          toast.success('Notification d\'accès renvoyée');
        } else {
          toast.warning('Notification non envoyée');
        }
      } else if (result.email_sent) {
        toast.success('Invitation renvoyée par email');
      } else {
        toast.warning('Email non envoyé — utilisez le lien ci-dessous.');
      }

      if (result.invite_link) {
        setLastInviteResult({
          email: result.user?.email || '',
          emailSent: result.email_sent,
          emailError: result.email_error,
          emailType: result.email_type,
          inviteLink: result.invite_link,
        });
      }
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors du renvoi de l\'invitation';
      toast.error(msg);
    }
  };

  const handleResetPassword = async (userId) => {
    try {
      const result = await resetPasswordMutation.mutateAsync(userId);
      if (result.temporary_credentials) {
        setCredentialsModal({
          ...result.temporary_credentials,
          credentialsShownOnce: result.credentials_shown_once,
        });
        toast.success('Mot de passe temporaire régénéré');
      }
    } catch (err) {
      const msg = err.response?.data?.error || 'Erreur lors de la réinitialisation';
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
            {showInvite ? <FaTimes /> : <FaUserPlus />}
            {showInvite ? ' Annuler' : ' Ajouter un collaborateur'}
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
          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            <button
              type="button"
              onClick={() => setInviteCreationMode('email')}
              style={{
                padding: '8px 16px',
                borderRadius: 6,
                border: inviteCreationMode === 'email' ? '2px solid #667eea' : '1px solid #ddd',
                background: inviteCreationMode === 'email' ? '#eef0ff' : '#fff',
                cursor: 'pointer',
                fontSize: 13,
                fontWeight: inviteCreationMode === 'email' ? 600 : 400,
              }}
            >
              <FaEnvelope style={{ marginRight: 6 }} />
              Par email
            </button>
            <button
              type="button"
              onClick={() => setInviteCreationMode('username')}
              style={{
                padding: '8px 16px',
                borderRadius: 6,
                border: inviteCreationMode === 'username' ? '2px solid #667eea' : '1px solid #ddd',
                background: inviteCreationMode === 'username' ? '#eef0ff' : '#fff',
                cursor: 'pointer',
                fontSize: 13,
                fontWeight: inviteCreationMode === 'username' ? 600 : 400,
              }}
            >
              <FaKey style={{ marginRight: 6 }} />
              Par identifiant
            </button>
          </div>

          {inviteCreationMode === 'email' ? (
            <div style={{ marginBottom: 12, fontSize: 13, color: '#666' }}>
              <FaEnvelope style={{ marginRight: 6, verticalAlign: 'middle' }} />
              Un email d'invitation sera envoyé avec un lien d'activation (valable 48h).
            </div>
          ) : (
            <div style={{ marginBottom: 12, fontSize: 13, color: '#666' }}>
              <FaKey style={{ marginRight: 6, verticalAlign: 'middle' }} />
              Création d'un compte avec identifiant et mot de passe temporaire (14 jours).
              {inviteUsername.trim() && (
                <span style={{ display: 'block', marginTop: 4, fontFamily: 'monospace', color: '#333' }}>
                  Aperçu : {inviteUsername.trim().toLowerCase()}
                </span>
              )}
            </div>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
            {inviteCreationMode === 'email' ? (
              <div>
                <label htmlFor="invite-email" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                  Email *
                </label>
                <input
                  id="invite-email"
                  name="email"
                  type="email"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  placeholder="email@exemple.ch"
                  autoComplete="off"
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
            ) : (
              <>
                <div>
                  <label htmlFor="invite-username" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                    Identifiant local *
                  </label>
                  <input
                    id="invite-username"
                    name="username"
                    type="text"
                    value={inviteUsername}
                    onChange={(e) => setInviteUsername(e.target.value.toLowerCase())}
                    placeholder="s.dupont"
                    autoComplete="off"
                    required
                    minLength={3}
                    pattern="[a-z0-9._-]+"
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
                  <label htmlFor="invite-contact-email" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                    Email de contact
                  </label>
                  <input
                    id="invite-contact-email"
                    name="contact_email"
                    type="email"
                    value={inviteEmail}
                    onChange={(e) => setInviteEmail(e.target.value)}
                    placeholder="prenom.nom@institution.ch"
                    autoComplete="off"
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
              </>
            )}
            <div>
              <label htmlFor="invite-role" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Rôle *
              </label>
              <ChipSelect
                id="invite-role"
                name="institution_role"
                ariaLabel="Rôle"
                block
                value={inviteRole}
                options={roleOptions}
                onChange={setInviteRole}
              />
              <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>
                {roleOptions.find((r) => r.value === inviteRole)?.desc || ''}
              </div>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
            <div>
              <label htmlFor="invite-first-name" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Prénom
              </label>
              <input
                id="invite-first-name"
                name="first_name"
                type="text"
                value={inviteFirstName}
                onChange={(e) => setInviteFirstName(e.target.value)}
                placeholder="Prénom"
                autoComplete="off"
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
              <label htmlFor="invite-last-name" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Nom
              </label>
              <input
                id="invite-last-name"
                name="last_name"
                type="text"
                value={inviteLastName}
                onChange={(e) => setInviteLastName(e.target.value)}
                placeholder="Nom"
                autoComplete="off"
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
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16, alignItems: 'start' }}>
            <div>
              <label htmlFor="invite-job-title" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Fonction / Métier
              </label>
              <JobTitleCombobox
                id="invite-job-title"
                value={inviteJobTitle}
                onChange={setInviteJobTitle}
                inputStyle={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #ddd',
                  borderRadius: 6,
                  fontSize: 14,
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div style={{ fontSize: 12, color: '#888', alignSelf: 'center', paddingTop: 18 }}>
              Choisissez une suggestion ou saisissez librement.
              <br />
              Information organisationnelle, sans impact sur les permissions.
            </div>
          </div>
          <button
            type="submit"
            className={styles.saveBtn}
            disabled={inviteMutation.isPending}
          >
            {inviteCreationMode === 'email' ? <FaEnvelope /> : <FaKey />}
            {inviteMutation.isPending
              ? ' En cours...'
              : inviteCreationMode === 'email'
                ? ' Envoyer l\'invitation'
                : ' Créer le compte'}
          </button>
        </form>
      )}

      {/* Credentials one-shot modal (Mode B) */}
      {credentialsModal && (
        <div className={styles.modal}>
          <div className={styles.modalContent} style={{ maxWidth: 480 }}>
            <div className={styles.modalHeader}>
              <h3>Identifiants de connexion</h3>
              <button onClick={() => setCredentialsModal(null)}>
                <FaTimes />
              </button>
            </div>
            <div className={styles.modalBody}>
              <p style={{ color: '#b71c1c', fontWeight: 600, marginBottom: 12 }}>
                ⚠️ Ces identifiants ne seront affichés qu'une seule fois. Notez-les maintenant.
              </p>
              <p style={{ fontSize: 13, color: '#555', marginBottom: 12 }}>
                Connexion avec : <strong>Identifiant</strong> (pas d'email)
              </p>
              <div style={{ background: '#f5f5f5', padding: 14, borderRadius: 8, fontFamily: 'monospace', fontSize: 13 }}>
                <div style={{ marginBottom: 8 }}>
                  <strong>Identifiant :</strong> {credentialsModal.username}
                  <button
                    type="button"
                    onClick={() => copyToClipboard(credentialsModal.username)}
                    style={{ marginLeft: 8, padding: '2px 8px', fontSize: 11, cursor: 'pointer' }}
                  >
                    <FaCopy />
                  </button>
                </div>
                <div>
                  <strong>Mot de passe temporaire :</strong> {credentialsModal.temporary_password}
                  <button
                    type="button"
                    onClick={() => copyToClipboard(credentialsModal.temporary_password)}
                    style={{ marginLeft: 8, padding: '2px 8px', fontSize: 11, cursor: 'pointer' }}
                  >
                    <FaCopy />
                  </button>
                </div>
              </div>
              <p style={{ fontSize: 12, color: '#666', marginTop: 12 }}>
                Le collaborateur devra changer son mot de passe à la première connexion.
              </p>
            </div>
            <div className={styles.modalActions}>
              <button
                onClick={() => setCredentialsModal(null)}
                style={{ backgroundColor: '#667eea', color: 'white' }}
              >
                J'ai noté les identifiants
              </button>
            </div>
          </div>
        </div>
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
                {lastInviteResult.emailType === 'access_notification'
                  ? (lastInviteResult.emailSent
                    ? '✅ Notification d\'accès envoyée'
                    : '⚠️ Notification d\'accès non envoyée')
                  : lastInviteResult.emailSent
                    ? `✅ Invitation envoyée${lastInviteResult.email ? ` à ${lastInviteResult.email}` : ''}`
                    : `⚠️ Email non envoyé${lastInviteResult.email ? ` à ${lastInviteResult.email}` : ''}`}
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

      {/* Collaborateurs en attente d'activation */}
      {canEdit && pendingUsers.length > 0 && (
        <div style={{
          marginBottom: 20,
          padding: 16,
          background: '#e3f2fd',
          border: '1px solid #90caf9',
          borderRadius: 8,
        }}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1565c0' }}>
            En attente d'activation ({pendingUsers.length})
          </h4>
          {pendingUsers.map((u) => (
            <div key={u.id} style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '8px 12px',
              background: '#fff',
              borderRadius: 6,
              marginBottom: 6,
              fontSize: 13,
            }}>
              <div style={{ flex: 1 }}>
                <strong>{u.first_name || u.last_name ? `${u.first_name || ''} ${u.last_name || ''}`.trim() : u.username}</strong>
                {u.username && (
                  <span style={{ marginLeft: 8, color: '#666', fontFamily: 'monospace', fontSize: 12 }}>
                    {u.username}
                  </span>
                )}
                <span style={{ marginLeft: 8, fontSize: 11, color: '#e65100' }}>
                  {PENDING_REASON_LABELS[u.pending_reason] || u.pending_reason}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                {u.authentication_method === 'username' && (
                  <button
                    onClick={() => handleResetPassword(u.id)}
                    disabled={resetPasswordMutation.isPending}
                    title="Réinitialiser le mot de passe"
                    style={actionBtnStyle('#1565c0', '#e3f2fd')}
                  >
                    <FaKey style={{ fontSize: 11 }} />
                  </button>
                )}
                {(u.account_status === 'invited' || u.account_status === 'expired') && (
                  <button
                    onClick={() => handleResendInvite(u.id)}
                    disabled={resendMutation.isPending}
                    title="Renvoyer l'invitation"
                    style={actionBtnStyle('#1565c0', '#e3f2fd')}
                  >
                    <FaRedo style={{ fontSize: 11 }} />
                  </button>
                )}
              </div>
            </div>
          ))}
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
              <th style={thStyle}>Fonction</th>
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
              const isUsernameAuth = user.authentication_method === 'username';

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
                    <div style={{ fontSize: 13, color: '#888' }}>
                      {user.email || user.username}
                    </div>
                    {user.authentication_method === 'username' && user.force_password_change && (
                      <span style={{
                        display: 'inline-block',
                        marginTop: 4,
                        padding: '2px 6px',
                        borderRadius: 4,
                        fontSize: 10,
                        background: '#fff3e0',
                        color: '#e65100',
                      }}>
                        {user.first_login_completed_at ? 'MDP temporaire' : 'Jamais connecté'}
                      </span>
                    )}
                  </td>
                  <td style={{ padding: '12px' }}>
                    {canEdit ? (
                      <JobTitleCombobox
                        id={`job-title-${user.id}`}
                        ariaLabel={`Fonction / métier de ${user.first_name || user.username || 'utilisateur'}`}
                        value={user.job_title || ''}
                        onCommit={(val) => handleJobTitleSave(user, val)}
                        disabled={updateProfileMutation.isPending}
                        placeholder="—"
                        inputStyle={{
                          width: '100%',
                          minWidth: 140,
                          padding: '6px 10px',
                          border: '1px solid #ddd',
                          borderRadius: 6,
                          fontSize: 13,
                          background: 'white',
                          boxSizing: 'border-box',
                        }}
                      />
                    ) : (
                      <span style={{ fontSize: 13, color: '#555' }}>
                        {user.job_title || '-'}
                      </span>
                    )}
                  </td>
                  <td style={{ padding: '12px' }}>
                    {canEdit && !isSelf && !isDisabled ? (
                      <ChipSelect
                        id={`role-${user.id}`}
                        name="institution_role"
                        ariaLabel={`Rôle de ${user.first_name || user.username || 'utilisateur'}`}
                        value={user.institution_role || ''}
                        options={roleOptions}
                        onChange={(val) => handleRoleChange(user.id, val)}
                        disabled={updateRoleMutation.isPending}
                      />
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
                        <button
                          onClick={() => openEditProfile(user)}
                          title="Modifier le profil"
                          style={actionBtnStyle('#5e35b1', '#ede7f6')}
                        >
                          <FaEdit style={{ fontSize: 11 }} />
                        </button>
                        {/* Renvoyer invitation - email mode only */}
                        {!isUsernameAuth && (isInvited || isDisabled) && (
                          <button
                            onClick={() => handleResendInvite(user.id)}
                            disabled={resendMutation.isPending}
                            title={isDisabled ? 'Réactiver et renvoyer invitation' : 'Renvoyer l\'invitation'}
                            style={actionBtnStyle('#1565c0', '#e3f2fd')}
                          >
                            <FaRedo style={{ fontSize: 11 }} />
                          </button>
                        )}
                        {/* Reset MDP - Mode B */}
                        {isUsernameAuth && !isSelf && !isDisabled && (
                          <button
                            onClick={() => handleResetPassword(user.id)}
                            disabled={resetPasswordMutation.isPending}
                            title="Réinitialiser le mot de passe temporaire"
                            style={actionBtnStyle('#1565c0', '#e3f2fd')}
                          >
                            <FaKey style={{ fontSize: 11 }} />
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

      {/* Modal édition profil */}
      {editProfileUser && (
        <div className={styles.modal}>
          <div className={styles.modalContent} style={{ maxWidth: 480 }}>
            <div className={styles.modalHeader}>
              <h3>Modifier le profil</h3>
              <button onClick={() => setEditProfileUser(null)}>
                <FaTimes />
              </button>
            </div>
            <form onSubmit={handleEditProfileSave}>
              <div className={styles.modalBody}>
                {editProfileUser.username && (
                  <p style={{ fontSize: 13, color: '#666', marginBottom: 12 }}>
                    Identifiant de connexion : <strong>{editProfileUser.username}</strong>
                  </p>
                )}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                  <div>
                    <label htmlFor="edit-first-name" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                      Prénom
                    </label>
                    <input
                      id="edit-first-name"
                      type="text"
                      value={editFirstName}
                      onChange={(e) => setEditFirstName(e.target.value)}
                      placeholder="Prénom"
                      autoComplete="off"
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
                    <label htmlFor="edit-last-name" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                      Nom
                    </label>
                    <input
                      id="edit-last-name"
                      type="text"
                      value={editLastName}
                      onChange={(e) => setEditLastName(e.target.value)}
                      placeholder="Nom"
                      autoComplete="off"
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
                <div>
                  <label htmlFor="edit-email" style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                    Email de contact
                  </label>
                  <input
                    id="edit-email"
                    type="email"
                    value={editEmail}
                    onChange={(e) => setEditEmail(e.target.value)}
                    placeholder="email@institution.ch"
                    autoComplete="off"
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
              <div className={styles.modalActions}>
                <button type="button" onClick={() => setEditProfileUser(null)}>Annuler</button>
                <button
                  type="submit"
                  style={{ backgroundColor: '#5e35b1', color: 'white' }}
                  disabled={updateProfileMutation.isPending}
                >
                  {updateProfileMutation.isPending ? 'Enregistrement...' : 'Enregistrer'}
                </button>
              </div>
            </form>
          </div>
        </div>
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
                    <strong>{confirmAction.user.email || confirmAction.user.username}</strong> ?
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
                    <strong>{confirmAction.user.email || confirmAction.user.username}</strong>{' '}
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
