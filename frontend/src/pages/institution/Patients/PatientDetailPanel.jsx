// pages/institution/Patients/PatientDetailPanel.jsx
/**
 * Panel latéral — fiche détaillée d'un patient.
 * Mode lecture + mode édition inline compact.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  FaUser, FaPhone, FaMapMarkerAlt, FaHome,
  FaTruck, FaShieldAlt, FaGavel, FaStickyNote, FaEdit, FaLock,
  FaEye, FaEyeSlash, FaEnvelope, FaCheck, FaTimes,
} from 'react-icons/fa';
import { FiChevronDown } from 'react-icons/fi';
import { HiOutlineX } from 'react-icons/hi';
import { toast } from 'sonner';
import {
  useInstitutionMe,
  useUpdatePatient,
  usePatientSyncStatus,
  usePatientMatches,
  useConfirmPatientMatch,
  useRejectPatientMatch,
  usePatientSuggestions,
  useConfirmPatientSuggestion,
  useRejectPatientSuggestion,
} from '../../../hooks/useInstitutionData';
import { canManageRequests, canViewAdminData, canEditPatientBillingData } from '../../../utils/institutionPermissions';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import s from './PatientDetailPanel.module.css';

const GENDER_LABELS = { HOMME: 'Monsieur', FEMME: 'Madame', AUTRE: 'Autre' };
const GENDER_OPTIONS = [
  { value: '', label: '— Genre —' },
  { value: 'HOMME', label: 'M.' },
  { value: 'FEMME', label: 'Mme' },
  { value: 'AUTRE', label: 'Autre' },
];

const GUARDIANSHIP_TYPE_LABELS = {
  opad: 'OPAD / SPAd',
  curatorship: 'Curateur professionnel',
  lawyer: 'Avocat',
  family: 'Famille',
  other: 'Autre',
};

const GUARDIANSHIP_TYPE_OPTIONS = [
  { value: '', label: '— Type —' },
  { value: 'opad', label: 'OPAD / SPAd' },
  { value: 'curatorship', label: 'Curateur pro.' },
  { value: 'lawyer', label: 'Avocat' },
  { value: 'family', label: 'Famille' },
  { value: 'other', label: 'Autre' },
];

const GUARDIANSHIP_TYPE_COLORS = {
  opad: { bg: '#EDE9FE', text: '#6D28D9', border: '#C4B5FD' },
  curatorship: { bg: '#FEF3C7', text: '#92400E', border: '#FDE68A' },
  lawyer: { bg: '#DBEAFE', text: '#1E40AF', border: '#93C5FD' },
  family: { bg: '#D1FAE5', text: '#065F46', border: '#6EE7B7' },
  other: { bg: '#F1F5F9', text: '#475569', border: '#CBD5E1' },
};

const capitalizeFirstName = (str) =>
  str.split(/([\s-])/).map((part) => {
    if (part === ' ' || part === '-') return part;
    return part.charAt(0).toUpperCase() + part.slice(1).toLowerCase();
  }).join('');

const fmtDate = (d) => {
  if (!d) return '—';
  return new Date(d).toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit', year: 'numeric' });
};

const getInitials = (first, last) => {
  const f = (first || '').charAt(0).toUpperCase();
  const l = (last || '').charAt(0).toUpperCase();
  return f + l || '?';
};

const age = (dob) => {
  if (!dob) return null;
  const born = new Date(dob);
  const now = new Date();
  let a = now.getFullYear() - born.getFullYear();
  if (now.getMonth() < born.getMonth() || (now.getMonth() === born.getMonth() && now.getDate() < born.getDate())) a--;
  return a;
};

const toInputDate = (d) => {
  if (!d) return '';
  try { return new Date(d).toISOString().split('T')[0]; } catch { return ''; }
};

const buildFormData = (p) => ({
  gender: p.gender || '',
  first_name: p.first_name || '',
  last_name: p.last_name || '',
  dob: toInputDate(p.dob),
  phone: p.phone || '',
  address: p.address || '',
  postal_code: p.postal_code || '',
  city: p.city || '',
  door_code: p.door_code || '',
  floor: p.floor || '',
  access_notes: p.access_notes || '',
  residence_name: p.residence_name || '',
  avs_number: p.avs_number || '',
  insurance_name: p.insurance_name || '',
  insurance_number: p.insurance_number || '',
  has_guardianship: !!p.has_guardianship,
  guardianship_type: p.guardianship_type || '',
  guardian_name: p.guardian_name || '',
  guardian_organization: p.guardian_organization || '',
  guardian_phone: p.guardian_phone || '',
  guardian_email: p.guardian_email || '',
  guardian_address: p.guardian_address || '',
  notes: p.notes || '',
});

// ─── Shared sub-components (OUTSIDE main component to avoid remount) ──

const ERow = ({ label, children }) => (
  <div className={s.eRow}>
    <span className={s.eLabel}>{label}</span>
    <div className={s.eField}>{children}</div>
  </div>
);

const VRow = ({ label, value, mono, icon }) => {
  if (!value) return null;
  return (
    <div className={s.infoRow}>
      {icon && <span className={s.infoIcon}>{icon}</span>}
      <span className={s.infoLabel}>{label}</span>
      <span className={`${s.infoValue} ${mono ? s.infoMono : ''}`}>{value}</span>
    </div>
  );
};

// ─── Sync status sub-component ──────────────────────────────────

const SyncStatusSection = ({ patientId, institutionRole: _institutionRole }) => {
  const { data: syncData, isLoading } = usePatientSyncStatus(patientId);

  if (isLoading || !syncData) return null;
  if (!syncData.synced) return null;

  const lastEvent = syncData.events?.[0];
  const statusColor = {
    success: '#059669',
    pending: '#d97706',
    processing: '#0284c7',
    partial_failure: '#dc2626',
    failed: '#dc2626',
  };

  return (
    <div className={s.section} style={{ borderLeft: '3px solid #7C3AED' }}>
      <div className={s.sectionHeader}>
        <FaShieldAlt className={s.sectionIcon} style={{ color: '#7C3AED' }} />
        <span>Synchronisation</span>
      </div>
      <div className={s.sectionBody}>
        {/* Source de vérité */}
        {syncData.source && (
          <div style={{ fontSize: 12, color: '#666', marginBottom: 6 }}>
            Source de vérité : <strong>{syncData.source.institution_name}</strong>
            {syncData.source.institution_type && (
              <span style={{
                marginLeft: 6,
                fontSize: 10,
                padding: '1px 6px',
                background: '#f5f0ff',
                color: '#7C3AED',
                borderRadius: 8,
              }}>
                {syncData.source.institution_type}
              </span>
            )}
          </div>
        )}

        {/* Identité */}
        {syncData.identity && (
          <div style={{ fontSize: 12, color: '#888', marginBottom: 6 }}>
            Identité : confiance <strong>{syncData.identity.confidence_level}</strong>
            {syncData.identity.avs_last4 && ` · AVS ***${syncData.identity.avs_last4}`}
            {` · ${syncData.identity.active_links_count} entité${syncData.identity.active_links_count !== 1 ? 's' : ''} liée${syncData.identity.active_links_count !== 1 ? 's' : ''}`}
          </div>
        )}

        {/* Dernière synchro */}
        {lastEvent && (
          <div style={{
            fontSize: 12,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 0',
          }}>
            <span style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: statusColor[lastEvent.status] || '#999',
              flexShrink: 0,
            }} />
            <span>
              Dernière synchro : <strong>{lastEvent.status}</strong>
              {lastEvent.processed_at && ` le ${new Date(lastEvent.processed_at).toLocaleString('fr-CH')}`}
              {lastEvent.retry_count > 0 && ` (${lastEvent.retry_count} tentative${lastEvent.retry_count > 1 ? 's' : ''})`}
            </span>
          </div>
        )}

        {/* Champs synchronisés */}
        {syncData.data_source_flags && Object.keys(syncData.data_source_flags).length > 0 && (
          <div style={{ marginTop: 6 }}>
            <span style={{ fontSize: 11, color: '#999' }}>Champs synchronisés :</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 3 }}>
              {Object.entries(syncData.data_source_flags).map(([field, source]) => (
                <span key={field} style={{
                  fontSize: 10,
                  padding: '1px 6px',
                  background: source === 'sync_curatelle' ? '#f5f0ff' : '#f0f0f0',
                  color: source === 'sync_curatelle' ? '#7C3AED' : '#666',
                  borderRadius: 8,
                }}>
                  {field}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ─── Matching suggestions sub-component ─────────────────────────

const MatchingSuggestions = ({ patientId }) => {
  const { data: matchData, isLoading } = usePatientMatches(patientId);
  const confirmMutation = useConfirmPatientMatch();
  const rejectMutation = useRejectPatientMatch();

  if (isLoading || !matchData) return null;
  if (!matchData.matches || matchData.matches.length === 0) return null;

  const handleConfirm = async (identityId) => {
    try {
      await confirmMutation.mutateAsync({ patientId, identityId });
      toast.success('Correspondance confirmée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };

  const handleReject = async (identityId) => {
    try {
      await rejectMutation.mutateAsync({ patientId, identityId });
      toast.success('Suggestion rejetée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };

  return (
    <div className={s.section} style={{ borderLeft: '3px solid #d97706' }}>
      <div className={s.sectionHeader}>
        <FaUser className={s.sectionIcon} style={{ color: '#d97706' }} />
        <span>Correspondances potentielles</span>
        <span style={{
          fontSize: 10,
          padding: '1px 6px',
          background: '#fef3c7',
          color: '#92400e',
          borderRadius: 8,
          marginLeft: 4,
        }}>
          {matchData.total}
        </span>
      </div>
      <div className={s.sectionBody}>
        <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>
          Ce patient n'a pas de numéro AVS. Des correspondances ont été trouvées
          par nom et date de naissance.
        </div>
        {matchData.matches.slice(0, 3).map((match, idx) => (
          <div key={idx} style={{
            padding: '8px 10px',
            border: '1px solid #e5e7eb',
            borderRadius: 6,
            marginBottom: 6,
            fontSize: 12,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong>Score : {match.match_score}%</strong>
                <span style={{ marginLeft: 8, color: '#888' }}>
                  {match.signals?.join(', ')}
                </span>
              </div>
              {match.type === 'identity' && (
                <span style={{
                  fontSize: 10,
                  padding: '1px 6px',
                  background: '#dbeafe',
                  color: '#1e40af',
                  borderRadius: 8,
                }}>
                  {match.linked_entities_count} entité{match.linked_entities_count !== 1 ? 's' : ''}
                </span>
              )}
            </div>
            {match.avs_last4 && (
              <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
                AVS : ***{match.avs_last4}
              </div>
            )}
            {match.institution_name && (
              <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
                Institution : {match.institution_name}
              </div>
            )}
            {match.identity_id && (
              <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                <button
                  onClick={() => handleConfirm(match.identity_id)}
                  disabled={confirmMutation.isPending}
                  style={{
                    padding: '3px 10px',
                    borderRadius: 4,
                    border: 'none',
                    background: '#059669',
                    color: '#fff',
                    fontSize: 11,
                    cursor: 'pointer',
                    fontWeight: 500,
                  }}
                >
                  <FaCheck size={9} /> Confirmer
                </button>
                <button
                  onClick={() => handleReject(match.identity_id)}
                  disabled={rejectMutation.isPending}
                  style={{
                    padding: '3px 10px',
                    borderRadius: 4,
                    border: '1px solid #ddd',
                    background: '#fff',
                    color: '#666',
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  <FaTimes size={9} /> Rejeter
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── Link suggestions sub-component ─────────────────────────────

const LinkSuggestions = ({ patientId }) => {
  const { data: suggestionsData, isLoading } = usePatientSuggestions(patientId);
  const confirmMutation = useConfirmPatientSuggestion();
  const rejectMutation = useRejectPatientSuggestion();

  if (isLoading || !suggestionsData) return null;
  if (!suggestionsData.suggestions || suggestionsData.suggestions.length === 0) return null;

  const handleConfirm = async (suggestionId) => {
    try {
      await confirmMutation.mutateAsync({ patientId, suggestionId });
      toast.success('Lien confirmé — synchronisation en cours');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la confirmation');
    }
  };

  const handleReject = async (suggestionId) => {
    try {
      await rejectMutation.mutateAsync({ patientId, suggestionId });
      toast.success('Suggestion rejetée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };

  return (
    <div className={s.section} style={{ borderLeft: '3px solid #d97706' }}>
      <div className={s.sectionHeader}>
        <FaGavel className={s.sectionIcon} style={{ color: '#d97706' }} />
        <span>Correspondances potentielles</span>
        <span style={{
          fontSize: 10,
          padding: '1px 6px',
          background: '#fef3c7',
          color: '#92400e',
          borderRadius: 8,
          marginLeft: 4,
        }}>
          {suggestionsData.total} en attente
        </span>
      </div>
      <div className={s.sectionBody}>
        <div style={{
          fontSize: 11,
          color: '#92400e',
          background: '#fffbeb',
          padding: '6px 10px',
          borderRadius: 6,
          marginBottom: 8,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}>
          <FaShieldAlt size={11} />
          Correspondance non confirmée — vérifiez avant de lier
        </div>
        {suggestionsData.suggestions.map((suggestion) => (
          <div key={suggestion.id} style={{
            padding: '10px',
            border: '1px solid #e5e7eb',
            borderRadius: 6,
            marginBottom: 6,
            fontSize: 12,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <div>
                <strong>Score : {suggestion.match_score}%</strong>
                <span style={{ marginLeft: 8, color: '#888', fontSize: 11 }}>
                  {suggestion.match_signals && Object.keys(suggestion.match_signals)
                    .filter(k => suggestion.match_signals[k])
                    .join(', ')}
                </span>
              </div>
              <span style={{
                fontSize: 10,
                padding: '1px 6px',
                background: '#fef3c7',
                color: '#92400e',
                borderRadius: 8,
              }}>
                En attente
              </span>
            </div>
            {suggestion.target_patient && (
              <div style={{
                background: '#f9fafb',
                padding: '6px 8px',
                borderRadius: 4,
                marginBottom: 6,
                fontSize: 11,
              }}>
                <div style={{ fontWeight: 600 }}>
                  {suggestion.target_patient.first_name} {suggestion.target_patient.last_name}
                </div>
                {suggestion.target_patient.dob && (
                  <div style={{ color: '#666' }}>
                    Né(e) le {new Date(suggestion.target_patient.dob).toLocaleDateString('fr-CH')}
                  </div>
                )}
                {suggestion.target_patient.city && (
                  <div style={{ color: '#666' }}>{suggestion.target_patient.city}</div>
                )}
                {suggestion.target_patient.institution_name && (
                  <div style={{ color: '#999', fontStyle: 'italic' }}>
                    {suggestion.target_patient.institution_name}
                  </div>
                )}
              </div>
            )}
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                onClick={() => handleConfirm(suggestion.id)}
                disabled={confirmMutation.isPending}
                style={{
                  padding: '4px 12px',
                  borderRadius: 4,
                  border: 'none',
                  background: '#059669',
                  color: '#fff',
                  fontSize: 11,
                  cursor: 'pointer',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                <FaCheck size={9} /> Confirmer le lien
              </button>
              <button
                onClick={() => handleReject(suggestion.id)}
                disabled={rejectMutation.isPending}
                style={{
                  padding: '4px 12px',
                  borderRadius: 4,
                  border: '1px solid #ddd',
                  background: '#fff',
                  color: '#666',
                  fontSize: 11,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                <FaTimes size={9} /> Rejeter
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── ChipDropdown (compact) ─────────────────────────────────────
const ChipDropdown = ({ value, options, onChange, className }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onClickOut = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onEsc = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClickOut);
    document.addEventListener('keydown', onEsc);
    return () => { document.removeEventListener('mousedown', onClickOut); document.removeEventListener('keydown', onEsc); };
  }, [open]);

  const selected = options.find((o) => String(o.value) === String(value));

  return (
    <div className={`${s.chipDrop} ${className || ''}`} ref={ref}>
      <button type="button" className={s.chipBtn} onClick={() => setOpen((p) => !p)}>
        <span className={s.chipText}>{selected?.label || '—'}</span>
        <FiChevronDown size={10} className={`${s.chipArrow} ${open ? s.chipArrowOpen : ''}`} />
      </button>
      {open && (
        <div className={s.chipMenu}>
          {options.filter(o => o.value).map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${s.chipOption} ${String(o.value) === String(value) ? s.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value === value ? '' : o.value); setOpen(false); }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Main component ────────────────────────────────────────────
const PatientDetailPanel = ({ patient, onClose }) => {
  const { data: meData } = useInstitutionMe();
  const updateMutation = useUpdatePatient();
  const institutionRole = meData?.institution_role;
  const canManage = canManageRequests(institutionRole);       // admin + requester : édition complète
  const canBillingEdit = canEditPatientBillingData(institutionRole); // billing : édition partielle
  const canSeeAdmin = canViewAdminData(institutionRole);
  const canEdit = canManage || canBillingEdit;                // peut afficher le bouton éditer
  const identityEditable = canManage;                         // seuls admin/requester modifient nom/genre/dob

  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState(() => buildFormData(patient));
  const [avsRevealed, setAvsRevealed] = useState(false);

  // Reset uniquement quand on change de patient (ID différent), pas sur chaque re-render
  const patientId = patient?.id;
  useEffect(() => {
    setForm(buildFormData(patient));
    setEditing(false);
    setAvsRevealed(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  const maskAvs = useCallback((avs) => {
    if (!avs) return '—';
    if (avs.length <= 4) return avs;
    return avs.substring(0, 4) + avs.substring(4).replace(/[0-9]/g, '*');
  }, []);

  const set = useCallback((field) => (e) => setForm(p => ({ ...p, [field]: e.target.value })), []);
  const setCheck = useCallback((field) => (e) => setForm(p => ({ ...p, [field]: e.target.checked })), []);

  const handleStartEdit = () => { setForm(buildFormData(patient)); setEditing(true); };
  const handleCancel = () => { setForm(buildFormData(patient)); setEditing(false); };
  const handleSave = async () => {
    try {
      // Nettoyer : convertir les chaînes vides en null pour éviter les erreurs de validation backend (regex phone, email)
      const cleaned = {};
      for (const [key, val] of Object.entries(form)) {
        cleaned[key] = (typeof val === 'string' && val.trim() === '') ? null : val;
      }

      // Si rôle billing (pas admin/requester), retirer les champs d'identité
      // Le backend filtre aussi côté serveur, mais on évite d'envoyer des champs inutiles
      if (!identityEditable) {
        delete cleaned.first_name;
        delete cleaned.last_name;
        delete cleaned.gender;
        delete cleaned.dob;
        delete cleaned.external_reference;
      }

      await updateMutation.mutateAsync({ patientId: patient.id, data: cleaned });
      toast.success('Patient mis à jour');
      setEditing(false);
    } catch (err) {
      const details = err?.response?.data?.details;
      const msg = details
        ? Object.values(details).flat().join(', ')
        : err?.response?.data?.error || 'Erreur lors de la mise à jour';
      toast.error(msg);
    }
  };

  if (!patient) return null;

  const fullAddr = [patient.address, patient.postal_code, patient.city].filter(Boolean).join(', ');
  const patientAge = age(patient.dob);
  const gColor = patient.guardianship_type ? GUARDIANSHIP_TYPE_COLORS[patient.guardianship_type] : null;

  return (
    <div className={s.panel}>
      {/* ── Header ── */}
      <div className={s.panelHeader}>
        <div className={s.headerInfo}>
          <div className={s.avatar}>
            {getInitials(editing ? form.first_name : patient.first_name, editing ? form.last_name : patient.last_name)}
          </div>
          <div className={s.headerMeta}>
            {editing && identityEditable ? (
              <div className={s.headerEditBlock}>
                <div className={s.headerEditRow}>
                  <input className={`${s.eInput} ${s.eInputUpper}`} value={form.last_name} onChange={(e) => setForm(p => ({ ...p, last_name: e.target.value.toUpperCase() }))} placeholder="Nom" />
                  <input className={s.eInput} value={form.first_name} onChange={(e) => setForm(p => ({ ...p, first_name: capitalizeFirstName(e.target.value) }))} placeholder="Prénom" />
                </div>
                <div className={s.headerEditRow}>
                  <ChipDropdown
                    value={form.gender}
                    options={GENDER_OPTIONS}
                    onChange={(v) => setForm(p => ({ ...p, gender: v }))}
                  />
                  <InlineDatePicker
                    value={form.dob}
                    onChange={(iso) => setForm(p => ({ ...p, dob: iso }))}
                    placeholder="Naissance"
                  />
                </div>
              </div>
            ) : (
              <>
                <span className={s.headerName}>{patient.last_name} {patient.first_name}</span>
                <span className={s.headerSub}>
                  {GENDER_LABELS[patient.gender] || ''}
                  {patient.dob && <> · {fmtDate(patient.dob)}{patientAge !== null && ` (${patientAge} ans)`}</>}
                </span>
              </>
            )}
          </div>
        </div>
        <div className={s.headerActions}>
          {canEdit && !editing && (
            <button className={s.editBtn} onClick={handleStartEdit} title="Modifier">
              <FaEdit size={12} />
            </button>
          )}
          <button className={s.closeBtn} onClick={onClose} aria-label="Fermer">
            <HiOutlineX />
          </button>
        </div>
      </div>

      {/* ── Body ── */}
      <div className={s.panelBody}>

        {/* ▸ Coordonnées */}
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <FaUser className={s.sectionIcon} />
            <span>Coordonnées</span>
          </div>
          <div className={s.sectionBody}>
            {editing ? (
              <>
                <ERow label="Tél."><input className={s.eInput} type="tel" value={form.phone} onChange={set('phone')} placeholder="+41 79..." /></ERow>
                <ERow label="Adresse">
                  <AddressAutocomplete
                    value={form.address}
                    onChange={(e) => setForm(p => ({ ...p, address: e.target.value }))}
                    onSelect={(item) => setForm(p => ({
                      ...p,
                      address: item.address || item.label || '',
                      postal_code: item.postcode || p.postal_code,
                      city: item.city || p.city,
                    }))}
                    placeholder="Rue, n°"
                    inputClassName={s.eInput}
                  />
                </ERow>
                <div className={s.eRowSplit}>
                  <ERow label="NPA"><input className={s.eInput} value={form.postal_code} onChange={set('postal_code')} placeholder="1200" /></ERow>
                  <ERow label="Ville"><input className={s.eInput} value={form.city} onChange={set('city')} placeholder="Genève" /></ERow>
                </div>
                <ERow label="Résidence"><input className={s.eInput} value={form.residence_name} onChange={set('residence_name')} placeholder="Nom EMS / résidence" /></ERow>
              </>
            ) : (
              <>
                <VRow label="Téléphone" value={patient.phone} icon={<FaPhone size={10} />} />
                <VRow label="Adresse" value={fullAddr} icon={<FaMapMarkerAlt size={10} />} />
                <VRow label="Résidence" value={patient.residence_name} icon={<FaHome size={10} />} />
                {!patient.phone && !fullAddr && !patient.residence_name && (
                  <div className={s.emptyHint}>Aucune coordonnée renseignée</div>
                )}
              </>
            )}
          </div>
        </div>

        {/* ▸ Accès & logistique */}
        {(editing || patient.door_code || patient.floor || patient.access_notes) && (
          <div className={s.section}>
            <div className={s.sectionHeader}>
              <FaTruck className={s.sectionIcon} />
              <span>Accès & logistique</span>
            </div>
            <div className={s.sectionBody}>
              {editing ? (
                <>
                  <div className={s.eRowSplit}>
                    <ERow label="Code"><input className={s.eInput} value={form.door_code} onChange={set('door_code')} placeholder="1234" /></ERow>
                    <ERow label="Étage"><input className={s.eInput} value={form.floor} onChange={set('floor')} placeholder="3e" /></ERow>
                  </div>
                  <textarea className={s.eTextarea} value={form.access_notes} onChange={set('access_notes')} placeholder="Indications d'accès..." rows={2} />
                </>
              ) : (
                <>
                  <VRow label="Code porte" value={patient.door_code} mono />
                  <VRow label="Étage" value={patient.floor} />
                  {patient.access_notes && <div className={s.noteBlock}>{patient.access_notes}</div>}
                </>
              )}
            </div>
          </div>
        )}

        {/* ▸ Administratif */}
        {canSeeAdmin && (editing || patient.avs_number || patient.insurance_name || patient.insurance_number) && (
          <div className={`${s.section} ${s.sectionRestricted}`}>
            <div className={s.sectionHeader}>
              <FaShieldAlt className={s.sectionIcon} />
              <span>Administratif</span>
              <span className={s.restrictedBadge}><FaLock size={8} /> Restreint</span>
            </div>
            <div className={s.sectionBody}>
              {editing ? (
                <>
                  <ERow label="N° AVS"><input className={s.eInput} value={form.avs_number} onChange={set('avs_number')} placeholder="756.xxxx.xxxx.xx" /></ERow>
                  <ERow label="Caisse"><input className={s.eInput} value={form.insurance_name} onChange={set('insurance_name')} placeholder="Nom caisse maladie" /></ERow>
                  <ERow label="N° assuré"><input className={s.eInput} value={form.insurance_number} onChange={set('insurance_number')} placeholder="N° police" /></ERow>
                </>
              ) : (
                <>
                  {patient.avs_number && (
                    <div className={s.infoRow}>
                      <span className={s.infoLabel}>N° AVS</span>
                      <span className={`${s.infoValue} ${s.infoMono}`}>
                        {avsRevealed ? patient.avs_number : maskAvs(patient.avs_number)}
                        <button type="button" className={s.avsToggle} onClick={() => setAvsRevealed(p => !p)} title={avsRevealed ? 'Masquer' : 'Afficher'}>
                          {avsRevealed ? <FaEyeSlash size={11} /> : <FaEye size={11} />}
                        </button>
                      </span>
                    </div>
                  )}
                  <VRow label="Caisse maladie" value={patient.insurance_name} />
                  <VRow label="N° assuré" value={patient.insurance_number} mono />
                </>
              )}
            </div>
          </div>
        )}

        {/* ▸ Curatelle */}
        {(editing || patient.has_guardianship) && (
          <div className={`${s.section} ${s.sectionCuratelle}`}>
            <div className={s.sectionHeader}>
              <FaGavel className={s.sectionIcon} />
              <span>Curatelle</span>
              {!editing && patient.guardianship_type && gColor && (
                <span className={s.guardianTypeBadge} style={{ background: gColor.bg, color: gColor.text, borderColor: gColor.border }}>
                  {GUARDIANSHIP_TYPE_LABELS[patient.guardianship_type]}
                </span>
              )}
            </div>
            <div className={s.sectionBody}>
              {editing && (
                <div className={s.eRow}>
                  <label className={s.eCheckLabel}>
                    <input type="checkbox" checked={form.has_guardianship} onChange={setCheck('has_guardianship')} />
                    <span>Sous curatelle</span>
                  </label>
                </div>
              )}
              {(editing ? form.has_guardianship : patient.has_guardianship) && (
                editing ? (
                  <>
                    <ERow label="Type">
                      <ChipDropdown
                        value={form.guardianship_type}
                        options={GUARDIANSHIP_TYPE_OPTIONS}
                        onChange={(v) => setForm(p => ({ ...p, guardianship_type: v }))}
                      />
                    </ERow>
                    <ERow label="Nom"><input className={s.eInput} value={form.guardian_name} onChange={set('guardian_name')} placeholder="Nom du curateur" /></ERow>
                    <ERow label="Org."><input className={s.eInput} value={form.guardian_organization} onChange={set('guardian_organization')} placeholder="Organisation" /></ERow>
                    <div className={s.eRowSplit}>
                      <ERow label="Tél."><input className={s.eInput} type="tel" value={form.guardian_phone} onChange={set('guardian_phone')} placeholder="+41..." /></ERow>
                      <ERow label="Email"><input className={s.eInput} type="email" value={form.guardian_email} onChange={set('guardian_email')} placeholder="email" /></ERow>
                    </div>
                    <ERow label="Adresse">
                      <AddressAutocomplete
                        value={form.guardian_address}
                        onChange={(e) => setForm(p => ({ ...p, guardian_address: e.target.value }))}
                        onSelect={(item) => setForm(p => ({ ...p, guardian_address: item.label || '' }))}
                        placeholder="Adresse curateur"
                        inputClassName={s.eInput}
                      />
                    </ERow>
                  </>
                ) : (
                  <>
                    <VRow label="Nom" value={patient.guardian_name} />
                    <VRow label="Organisation" value={patient.guardian_organization} />
                    <VRow label="Tél." value={patient.guardian_phone} icon={<FaPhone size={10} />} />
                    <VRow label="Email" value={patient.guardian_email} icon={<FaEnvelope size={10} />} />
                    <VRow label="Adresse" value={patient.guardian_address} icon={<FaMapMarkerAlt size={10} />} />
                  </>
                )
              )}
            </div>
          </div>
        )}

        {/* ▸ Notes */}
        {(editing || patient.notes) && (
          <div className={s.section}>
            <div className={s.sectionHeader}>
              <FaStickyNote className={s.sectionIcon} />
              <span>Notes internes</span>
            </div>
            <div className={s.sectionBody}>
              {editing ? (
                <textarea className={s.eTextarea} value={form.notes} onChange={set('notes')} placeholder="Notes internes..." rows={3} />
              ) : (
                <div className={s.noteBlock}>{patient.notes}</div>
              )}
            </div>
          </div>
        )}

        {/* ▸ Synchronisation cross-plateforme */}
        {!editing && <SyncStatusSection patientId={patient.id} institutionRole={institutionRole} />}

        {/* ▸ Link suggestions (confirmations en attente) */}
        {!editing && (
          <LinkSuggestions patientId={patient.id} />
        )}
        {/* ▸ Correspondances potentielles (si pas d'AVS, ancien système) */}
        {!editing && !patient.avs_number && (
          <MatchingSuggestions patientId={patient.id} />
        )}
      </div>

      {/* ── Footer (mode édition) ── */}
      {editing && (
        <div className={s.panelFooter}>
          <button className={s.cancelBtn} onClick={handleCancel} type="button">
            <FaTimes size={10} /> Annuler
          </button>
          <button className={s.saveBtn} onClick={handleSave} disabled={updateMutation.isPending} type="button">
            <FaCheck size={10} /> {updateMutation.isPending ? 'Sauvegarde...' : 'Enregistrer'}
          </button>
        </div>
      )}
    </div>
  );
};

export default PatientDetailPanel;
