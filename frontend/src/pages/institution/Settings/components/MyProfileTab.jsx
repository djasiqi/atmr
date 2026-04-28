// pages/institution/Settings/components/MyProfileTab.jsx
/**
 * Onglet "Mon profil" : chaque utilisateur peut gérer ses propres informations
 * et envoyer des demandes de droits à l'admin.
 *
 * Accessible par tous les rôles institution.
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { FaSave, FaUser, FaPaperPlane, FaClock, FaCheckCircle, FaTimesCircle, FaShieldAlt } from 'react-icons/fa';
import {
  useInstitutionMe,
  useUpdateMyProfile,
  usePermissionRequests,
  useCreatePermissionRequest,
} from '../../../../hooks/useInstitutionData';
import { isAdmin, getRoleLabel, getRoleBadgeColor } from '../../../../utils/institutionPermissions';
import { getAuthEnv } from '../../../../utils/webAuthSession';
import { toast } from 'sonner';
import styles from '../InstitutionSettings.module.css';

const ROLE_OPTIONS = [
  { value: 'institution_admin', label: 'Administrateur', desc: 'Accès complet à tous les paramètres et fonctionnalités' },
  { value: 'institution_requester', label: 'Demandeur', desc: 'Créer et gérer des demandes de transport, gérer les patients' },
  { value: 'institution_billing', label: 'Facturation', desc: 'Gérer les paramètres de facturation et les données administratives des patients' },
  { value: 'institution_reader', label: 'Lecteur', desc: 'Consultation uniquement (lecture seule)' },
];

const REQUEST_STATUS_CONFIG = {
  pending: { label: 'En attente', icon: FaClock, color: '#e65100', bg: '#fff3e0' },
  approved: { label: 'Approuvée', icon: FaCheckCircle, color: '#2e7d32', bg: '#e8f5e9' },
  denied: { label: 'Refusée', icon: FaTimesCircle, color: '#c62828', bg: '#ffebee' },
};

const MyProfileTab = () => {
  const { data: meData, isLoading } = useInstitutionMe();
  const updateProfileMutation = useUpdateMyProfile();
  const { data: permRequestsData } = usePermissionRequests();
  const createPermRequestMutation = useCreatePermissionRequest();

  const user = meData?.user;
  const isDemoEnv = (() => {
    try {
      return getAuthEnv() === 'demo';
    } catch {
      return false;
    }
  })();
  const institutionRole =
    meData?.institution_role || (isDemoEnv ? 'institution_admin' : undefined);
  const canRequestPerms = !isAdmin(institutionRole);

  const [form, setForm] = useState({
    first_name: '',
    last_name: '',
    phone: '',
  });

  // Permission request form
  const [showRequestForm, setShowRequestForm] = useState(false);
  const [requestedRole, setRequestedRole] = useState('');
  const [requestMessage, setRequestMessage] = useState('');

  const [phoneTouched, setPhoneTouched] = useState(false);

  useEffect(() => {
    if (user) {
      setForm({
        first_name: user.first_name || '',
        last_name: user.last_name || '',
        phone: user.phone || '',
      });
    }
  }, [user]);

  // ── Téléphone : formatage + validation ──
  // ── Indicatifs internationaux connus (pays pertinents) ──
  const COUNTRY_CODES = useMemo(() => [
    { code: '+41',  flag: '🇨🇭', name: 'Suisse',      maxLocal: 9,  format: [2, 3, 2, 2] },
    { code: '+33',  flag: '🇫🇷', name: 'France',      maxLocal: 9,  format: [1, 2, 2, 2, 2] },
    { code: '+49',  flag: '🇩🇪', name: 'Allemagne',   maxLocal: 11, format: [3, 4, 4] },
    { code: '+39',  flag: '🇮🇹', name: 'Italie',      maxLocal: 10, format: [3, 3, 4] },
    { code: '+43',  flag: '🇦🇹', name: 'Autriche',    maxLocal: 10, format: [3, 3, 4] },
    { code: '+44',  flag: '🇬🇧', name: 'Royaume-Uni', maxLocal: 10, format: [4, 3, 3] },
    { code: '+34',  flag: '🇪🇸', name: 'Espagne',     maxLocal: 9,  format: [3, 3, 3] },
    { code: '+351', flag: '🇵🇹', name: 'Portugal',    maxLocal: 9,  format: [3, 3, 3] },
    { code: '+352', flag: '🇱🇺', name: 'Luxembourg',  maxLocal: 9,  format: [3, 3, 3] },
    { code: '+32',  flag: '🇧🇪', name: 'Belgique',    maxLocal: 9,  format: [3, 2, 2, 2] },
    { code: '+1',   flag: '🇺🇸', name: 'USA/Canada',  maxLocal: 10, format: [3, 3, 4] },
    { code: '+90',  flag: '🇹🇷', name: 'Turquie',     maxLocal: 10, format: [3, 3, 4] },
    { code: '+383', flag: '🇽🇰', name: 'Kosovo',      maxLocal: 8,  format: [2, 3, 3] },
    { code: '+355', flag: '🇦🇱', name: 'Albanie',     maxLocal: 9,  format: [2, 3, 4] },
    { code: '+381', flag: '🇷🇸', name: 'Serbie',      maxLocal: 9,  format: [2, 3, 4] },
    { code: '+389', flag: '🇲🇰', name: 'Macédoine',   maxLocal: 8,  format: [2, 3, 3] },
  ], []);

  /** Détecte le pays à partir de l'indicatif */
  const detectCountry = useCallback((digits) => {
    // Trier par longueur de code décroissante pour matcher +383 avant +38
    const sorted = [...COUNTRY_CODES].sort((a, b) => b.code.length - a.code.length);
    for (const c of sorted) {
      if (digits.startsWith(c.code)) return c;
    }
    return null;
  }, [COUNTRY_CODES]);

  /** Normalise un numéro brut en ajoutant le + si nécessaire */
  const normalizeDigits = useCallback((raw) => {
    // Retirer tout sauf chiffres et +
    let digits = raw.replace(/[^\d+]/g, '');
    // Ne garder le + qu'au début
    if (digits.includes('+')) {
      digits = '+' + digits.replace(/\+/g, '');
    }

    // 0XX... → +41XX... (numéro local suisse)
    if (digits.startsWith('0') && !digits.startsWith('00')) {
      return '+41' + digits.substring(1);
    }
    // 00XX... → +XX... (double zéro international)
    if (digits.startsWith('00')) {
      return '+' + digits.substring(2);
    }
    // Déjà un + → vérifier si l'indicatif est reconnu
    if (digits.startsWith('+')) {
      const matched = detectCountry(digits);
      if (matched) return digits;
      // + mais indicatif inconnu → vérifier si c'est un numéro suisse local avec + collé
      const withoutPlus = digits.substring(1);
      if (/^[2-9]\d{8}$/.test(withoutPlus)) {
        return '+41' + withoutPlus;
      }
      return digits;
    }
    // Que des chiffres, pas de + → essayer de deviner l'indicatif
    // Tester avec + devant pour voir si un pays est reconnu (41xxx, 33xxx, etc.)
    const withPlus = '+' + digits;
    const guessed = detectCountry(withPlus);
    if (guessed) {
      return withPlus;
    }
    // 9 chiffres commençant par 2-9 = numéro suisse sans le 0 (ex: 762034041)
    if (/^[2-9]\d{8}$/.test(digits)) {
      return '+41' + digits;
    }
    // Pas reconnu → retourner tel quel avec + ajouté
    return withPlus;
  }, [detectCountry]);

  /** Formate un numéro selon les règles du pays détecté */
  const formatPhone = useCallback((raw) => {
    const digits = normalizeDigits(raw);

    const country = detectCountry(digits);
    if (!country) return digits; // Pas d'indicatif reconnu → pas de formatage

    const prefix = country.code;
    const local = digits.substring(prefix.length);

    // Limiter la longueur du numéro local
    const trimmedLocal = local.substring(0, country.maxLocal);

    // Appliquer le pattern de formatage
    let formatted = prefix;
    let pos = 0;
    for (const groupSize of country.format) {
      if (pos >= trimmedLocal.length) break;
      formatted += ' ' + trimmedLocal.substring(pos, pos + groupSize);
      pos += groupSize;
    }
    // Chiffres restants non couverts par le pattern
    if (pos < trimmedLocal.length) {
      formatted += ' ' + trimmedLocal.substring(pos);
    }

    return formatted;
  }, [detectCountry, normalizeDigits]);

  /** Détecte le pays actuel du numéro dans le formulaire */
  const detectedCountry = useMemo(() => {
    const raw = (form.phone || '').replace(/\s/g, '');
    if (!raw) return null;
    const normalized = normalizeDigits(raw);
    return detectCountry(normalized);
  }, [form.phone, detectCountry, normalizeDigits]);

  const phoneValidation = useMemo(() => {
    const phone = form.phone?.trim() || '';
    if (!phone) return { valid: true, touched: false, message: '', country: null };
    const clean = normalizeDigits(phone.replace(/\s/g, ''));

    const country = detectCountry(clean);

    if (!country) {
      // Indicatif inconnu → vérification basique
      const numPart = clean.replace(/^\+/, '');
      if (!/^\d+$/.test(numPart)) {
        return { valid: false, touched: phoneTouched, message: 'Caractères non autorisés', country: null };
      }
      if (numPart.length < 7) {
        return { valid: false, touched: phoneTouched, message: 'Numéro trop court', country: null };
      }
      if (numPart.length > 15) {
        return { valid: false, touched: phoneTouched, message: 'Numéro trop long', country: null };
      }
      return { valid: true, touched: phoneTouched, message: '', country: null };
    }

    // Pays détecté → vérification spécifique
    const local = clean.substring(country.code.length);
    if (!/^\d*$/.test(local)) {
      return { valid: false, touched: phoneTouched, message: 'Caractères non autorisés', country };
    }

    if (local.length < country.maxLocal) {
      return {
        valid: false, touched: phoneTouched,
        message: `Incomplet (${local.length}/${country.maxLocal} chiffres)`,
        country,
      };
    }
    if (local.length > country.maxLocal) {
      return {
        valid: false, touched: phoneTouched,
        message: `Trop long (max ${country.maxLocal} chiffres après ${country.code})`,
        country,
      };
    }

    return { valid: true, touched: phoneTouched, message: '', country };
  }, [form.phone, phoneTouched, detectCountry, normalizeDigits]);

  const handlePhoneChange = useCallback((e) => {
    const raw = e.target.value;
    setPhoneTouched(true);
    if (!raw.trim()) {
      setForm(p => ({ ...p, phone: '' }));
      return;
    }
    const formatted = formatPhone(raw);
    setForm(p => ({ ...p, phone: formatted }));
  }, [formatPhone]);

  const isDirty = useMemo(() => {
    if (!user) return false;
    return (
      (form.first_name?.trim() || '') !== (user.first_name || '') ||
      (form.last_name?.trim() || '') !== (user.last_name || '') ||
      (form.phone?.trim() || '') !== (user.phone || '')
    );
  }, [form, user]);

  const handleSave = async () => {
    // Valider le téléphone avant d'envoyer
    if (form.phone && !phoneValidation.valid) {
      toast.error('Veuillez corriger le numéro de téléphone');
      setPhoneTouched(true);
      return;
    }

    const payload = {};
    if ((form.first_name?.trim() || '') !== (user?.first_name || '')) {
      payload.first_name = form.first_name.trim() || null;
    }
    if ((form.last_name?.trim() || '') !== (user?.last_name || '')) {
      payload.last_name = form.last_name.trim() || null;
    }
    // Envoyer le téléphone sans espaces (format API: +41XXXXXXXXX)
    const cleanPhone = form.phone?.replace(/\s/g, '') || '';
    if (cleanPhone !== (user?.phone || '')) {
      payload.phone = cleanPhone || null;
    }

    if (Object.keys(payload).length === 0) {
      toast.info('Aucune modification détectée');
      return;
    }

    try {
      await updateProfileMutation.mutateAsync(payload);
      toast.success('Profil mis à jour');
    } catch (err) {
      const msg = err?.response?.data?.error || err?.response?.data?.details || 'Erreur lors de la mise à jour';
      toast.error(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
  };

  const handleSubmitPermRequest = async (e) => {
    e.preventDefault();
    if (!requestedRole) {
      toast.error('Veuillez sélectionner un rôle');
      return;
    }
    if (!requestMessage.trim() || requestMessage.trim().length < 5) {
      toast.error('Le message doit contenir au moins 5 caractères');
      return;
    }

    try {
      await createPermRequestMutation.mutateAsync({
        requested_role: requestedRole,
        message: requestMessage.trim(),
      });
      toast.success('Demande envoyée à l\'administrateur');
      setShowRequestForm(false);
      setRequestedRole('');
      setRequestMessage('');
    } catch (err) {
      const msg = err?.response?.data?.error || 'Erreur lors de l\'envoi';
      toast.error(msg);
    }
  };

  const myRequests = permRequestsData?.requests || [];
  const hasPending = myRequests.some(r => r.status === 'pending');

  if (isLoading) return <p>Chargement...</p>;

  return (
    <div className={styles.section}>
      {/* ── Informations personnelles ── */}
      <div className={styles.sectionHeader}>
        <h3><FaUser style={{ marginRight: 8 }} /> Mon profil</h3>
        <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
          Gérez vos informations personnelles. Ces données sont visibles par les
          administrateurs de votre institution.
        </p>
      </div>

      {/* Rôle actuel */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        marginBottom: 20,
        padding: '10px 16px',
        background: '#f8f9fa',
        borderRadius: 8,
        border: '1px solid #eee',
      }}>
        <FaShieldAlt style={{ color: getRoleBadgeColor(institutionRole), fontSize: 16 }} />
        <div>
          <div style={{ fontSize: 13, fontWeight: 500, color: '#333' }}>
            Rôle actuel : <span style={{
              display: 'inline-block',
              padding: '2px 10px',
              borderRadius: 10,
              fontSize: 12,
              fontWeight: 600,
              backgroundColor: `${getRoleBadgeColor(institutionRole)}15`,
              color: getRoleBadgeColor(institutionRole),
              marginLeft: 4,
            }}>
              {getRoleLabel(institutionRole)}
            </span>
          </div>
          <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>
            {ROLE_OPTIONS.find(r => r.value === institutionRole)?.desc || ''}
          </div>
        </div>
      </div>

      {/* Email (lecture seule) */}
      <div className={styles.profileForm}>
        <div className={styles.field}>
          <label>Email</label>
          <input
            type="email"
            value={user?.email || ''}
            disabled
            className={styles.readonlyField}
          />
          <span className={styles.fieldHint}>
            L'email ne peut pas être modifié. Contactez un administrateur si nécessaire.
          </span>
        </div>

        {/* Prénom + Nom */}
        <div className={styles.fieldRow}>
          <div className={styles.field}>
            <label>Prénom</label>
            <input
              type="text"
              value={form.first_name}
              onChange={(e) => setForm(p => ({ ...p, first_name: e.target.value }))}
              placeholder="Votre prénom"
            />
          </div>
          <div className={styles.field}>
            <label>Nom</label>
            <input
              type="text"
              value={form.last_name}
              onChange={(e) => setForm(p => ({ ...p, last_name: e.target.value }))}
              placeholder="Votre nom"
            />
          </div>
        </div>

        {/* Téléphone */}
        <div className={styles.field}>
          <label>Téléphone</label>
          <input
            type="tel"
            value={form.phone}
            onChange={handlePhoneChange}
            placeholder="+41 79 123 45 67"
          />
          {form.phone ? (
            <span className={styles.fieldHint}>
              {detectedCountry
                ? `${detectedCountry.name} (${detectedCountry.code})`
                : 'Indicatif non reconnu'}
              {phoneValidation.valid && detectedCountry ? ' · valide' : ''}
              {phoneTouched && !phoneValidation.valid && phoneValidation.message
                ? ` · ${phoneValidation.message}`
                : ''}
            </span>
          ) : (
            <span className={styles.fieldHint}>
              Format international, ex : +41 79 123 45 67
            </span>
          )}
        </div>

        {/* Bouton Enregistrer */}
        <button
          className={styles.saveBtn}
          onClick={handleSave}
          disabled={updateProfileMutation.isPending || !isDirty}
        >
          <FaSave /> {updateProfileMutation.isPending ? 'Enregistrement...' : 'Enregistrer'}
        </button>
      </div>

      {/* ── Demande de droits (non-admin uniquement) ── */}
      {canRequestPerms && (
        <>
          <hr style={{ border: 'none', borderTop: '1px solid #eee', margin: '28px 0' }} />

          <div className={styles.sectionHeader}>
            <h3><FaShieldAlt style={{ marginRight: 8 }} /> Demande de droits</h3>
            <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
              Vous pouvez demander un changement de rôle à l'administrateur de votre institution.
              Votre demande sera examinée et vous serez notifié de la décision.
            </p>
          </div>

          {/* Historique des demandes */}
          {myRequests.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <h4 style={{ fontSize: 14, color: '#333', marginBottom: 8 }}>Mes demandes</h4>
              {myRequests.map((req) => {
                const cfg = REQUEST_STATUS_CONFIG[req.status] || REQUEST_STATUS_CONFIG.pending;
                const StatusIcon = cfg.icon;
                return (
                  <div key={req.id} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '10px 14px',
                    background: cfg.bg,
                    borderRadius: 8,
                    marginBottom: 8,
                    border: `1px solid ${cfg.color}22`,
                  }}>
                    <StatusIcon style={{ color: cfg.color, fontSize: 16, flexShrink: 0 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 13, fontWeight: 500, color: '#333' }}>
                        Rôle demandé : {getRoleLabel(req.requested_role)}
                      </div>
                      <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>
                        {req.message}
                      </div>
                      <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
                        {new Date(req.created_at).toLocaleDateString('fr-CH', {
                          day: '2-digit',
                          month: '2-digit',
                          year: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </div>
                    </div>
                    <span style={{
                      padding: '3px 10px',
                      borderRadius: 10,
                      fontSize: 11,
                      fontWeight: 600,
                      color: cfg.color,
                      background: '#fff',
                      border: `1px solid ${cfg.color}33`,
                    }}>
                      {cfg.label}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Bouton + formulaire */}
          {!hasPending && !showRequestForm && (
            <button
              className={styles.saveBtn}
              onClick={() => setShowRequestForm(true)}
              style={{ width: 'auto' }}
            >
              <FaPaperPlane /> Demander un changement de rôle
            </button>
          )}

          {hasPending && !showRequestForm && (
            <div style={{
              padding: '10px 16px',
              background: '#fff3e0',
              border: '1px solid #ffe082',
              borderRadius: 8,
              fontSize: 13,
              color: '#5d4037',
            }}>
              <FaClock style={{ marginRight: 6, verticalAlign: 'middle' }} />
              Vous avez déjà une demande en attente. Veuillez patienter.
            </div>
          )}

          {showRequestForm && (
            <form onSubmit={handleSubmitPermRequest} style={{
              background: '#fafafa',
              padding: 20,
              borderRadius: 8,
              border: '1px solid #e0e0e0',
            }}>
              <div className={styles.field} style={{ marginBottom: 14 }}>
                <label>Rôle souhaité *</label>
                <select
                  value={requestedRole}
                  onChange={(e) => setRequestedRole(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #ddd',
                    borderRadius: 6,
                    fontSize: 14,
                    boxSizing: 'border-box',
                  }}
                >
                  <option value="">— Sélectionner —</option>
                  {ROLE_OPTIONS
                    .filter(r => r.value !== institutionRole) // Exclure le rôle actuel
                    .map(r => (
                      <option key={r.value} value={r.value}>{r.label}</option>
                    ))}
                </select>
                {requestedRole && (
                  <span className={styles.fieldHint}>
                    {ROLE_OPTIONS.find(r => r.value === requestedRole)?.desc || ''}
                  </span>
                )}
              </div>

              <div className={styles.field} style={{ marginBottom: 14 }}>
                <label>Justification *</label>
                <textarea
                  value={requestMessage}
                  onChange={(e) => setRequestMessage(e.target.value)}
                  placeholder="Expliquez pourquoi vous avez besoin de ce rôle (5-500 caractères)..."
                  rows={3}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #ddd',
                    borderRadius: 6,
                    fontSize: 14,
                    boxSizing: 'border-box',
                    resize: 'vertical',
                  }}
                />
                <span className={styles.fieldHint}>
                  {requestMessage.length}/500 caractères
                </span>
              </div>

              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  type="submit"
                  className={styles.saveBtn}
                  disabled={createPermRequestMutation.isPending}
                  style={{ width: 'auto' }}
                >
                  <FaPaperPlane />
                  {createPermRequestMutation.isPending ? ' Envoi...' : ' Envoyer la demande'}
                </button>
                <button
                  type="button"
                  onClick={() => { setShowRequestForm(false); setRequestedRole(''); setRequestMessage(''); }}
                  style={{
                    padding: '8px 16px',
                    background: 'white',
                    border: '1px solid #ddd',
                    borderRadius: 6,
                    cursor: 'pointer',
                    fontSize: 14,
                  }}
                >
                  Annuler
                </button>
              </div>
            </form>
          )}
        </>
      )}
    </div>
  );
};

export default MyProfileTab;
