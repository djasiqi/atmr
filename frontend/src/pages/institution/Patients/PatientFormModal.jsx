import React, { useState, useCallback, useRef, useEffect } from 'react';
import { FaTimes, FaCheck, FaPhone, FaMapMarkerAlt, FaUser, FaTruck, FaLock, FaStickyNote, FaShieldAlt, FaGavel, FaEnvelope, FaEye, FaEyeSlash } from 'react-icons/fa';
import { FiChevronDown } from 'react-icons/fi';
import { useCreatePatient, useUpdatePatient, useInstitutionMe } from '../../../hooks/useInstitutionData';
import { canViewAdminData, canEditAdminData, isCurator } from '../../../utils/institutionPermissions';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import { toast } from 'sonner';
import s from './InstitutionPatients.module.css';

const GENDER_OPTIONS = [
  { value: '', label: '— Civilité —' },
  { value: 'HOMME', label: 'Monsieur' },
  { value: 'FEMME', label: 'Madame' },
  { value: 'AUTRE', label: 'Autre' },
];

const GENDER_SHORT = { HOMME: 'M.', FEMME: 'Mme', AUTRE: '' };

const GUARDIANSHIP_TYPES = [
  { value: '', label: '— Sélectionner le type —' },
  { value: 'opad', label: 'OPAD / SPAd (Service de protection)' },
  { value: 'curatorship', label: 'Curateur professionnel' },
  { value: 'lawyer', label: 'Avocat / Étude juridique' },
  { value: 'family', label: 'Membre de la famille' },
  { value: 'other', label: 'Autre' },
];

const GUARDIANSHIP_TYPE_LABELS = {
  opad: 'OPAD / SPAd',
  curatorship: 'Curateur professionnel',
  lawyer: 'Avocat',
  family: 'Famille',
  other: 'Autre',
};

const GUARDIANSHIP_TYPE_COLORS = {
  opad: { bg: '#EDE9FE', text: '#6D28D9', border: '#C4B5FD' },
  curatorship: { bg: '#FEF3C7', text: '#92400E', border: '#FDE68A' },
  lawyer: { bg: '#DBEAFE', text: '#1E40AF', border: '#93C5FD' },
  family: { bg: '#D1FAE5', text: '#065F46', border: '#6EE7B7' },
  other: { bg: '#F1F5F9', text: '#475569', border: '#CBD5E1' },
};

const EMPTY_FORM = {
  gender: '', first_name: '', last_name: '', dob: '', phone: '',
  address: '', postal_code: '', city: '',
  door_code: '', floor: '', access_notes: '', residence_name: '',
  avs_number: '', insurance_name: '', insurance_number: '',
  has_guardianship: false, guardianship_type: '',
  guardian_name: '', guardian_organization: '',
  guardian_phone: '', guardian_email: '', guardian_address: '',
  notes: '',
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

const ChipDropdown = ({ value, options, onChange, disabled }) => {
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
    <div className={s.chipDrop} ref={ref}>
      <button type="button" className={s.chipBtn} onClick={() => !disabled && setOpen((p) => !p)} disabled={disabled}>
        <span className={s.chipText}>{selected?.label || '—'}</span>
        <FiChevronDown size={11} className={`${s.chipArrow} ${open ? s.chipArrowOpen : ''}`} />
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

/**
 * Shared patient creation/edit modal.
 * Used by InstitutionPatients (list page) and InstitutionRequestCreate (quick create).
 *
 * @param {function} onClose - close modal
 * @param {function} onSaved - called with the created/updated patient result
 * @param {object|null} editingPatient - patient to edit, or null for creation
 * @param {object} [initialFormOverrides] - optional form field overrides
 */
export default function PatientFormModal({ onClose, onSaved, editingPatient = null, initialFormOverrides }) {
  const createMutation = useCreatePatient();
  const updateMutation = useUpdatePatient();
  const { data: meData } = useInstitutionMe();

  const institutionRole = meData?.institution_role;
  const canSeeAdmin = canViewAdminData(institutionRole);
  const canEditAdmin = canEditAdminData(institutionRole);
  const isCuratelleInstitution = meData?.institution_type?.toLowerCase() === 'curatelle';
  const userIsCurator = isCurator(institutionRole);

  const buildInitialForm = useCallback(() => {
    if (editingPatient) {
      return {
        gender: editingPatient.gender || '',
        first_name: editingPatient.first_name || '',
        last_name: editingPatient.last_name || '',
        dob: editingPatient.dob ? editingPatient.dob.split('T')[0] : '',
        phone: editingPatient.phone || '',
        address: editingPatient.address || '',
        postal_code: editingPatient.postal_code || '',
        city: editingPatient.city || '',
        door_code: editingPatient.door_code || '',
        floor: editingPatient.floor || '',
        access_notes: editingPatient.access_notes || '',
        residence_name: editingPatient.residence_name || '',
        avs_number: editingPatient.avs_number || '',
        insurance_name: editingPatient.insurance_name || '',
        insurance_number: editingPatient.insurance_number || '',
        has_guardianship: editingPatient.has_guardianship || false,
        guardianship_type: editingPatient.guardianship_type || '',
        guardian_name: editingPatient.guardian_name || '',
        guardian_organization: editingPatient.guardian_organization || '',
        guardian_phone: editingPatient.guardian_phone || '',
        guardian_email: editingPatient.guardian_email || '',
        guardian_address: editingPatient.guardian_address || '',
        notes: editingPatient.notes || '',
      };
    }
    const base = { ...EMPTY_FORM, ...(initialFormOverrides || {}) };
    if (isCuratelleInstitution || userIsCurator) {
      const user = meData?.user;
      const instName = meData?.name || '';
      base.has_guardianship = true;
      base.guardianship_type = 'opad';
      base.guardian_name = user ? `${user.first_name || ''} ${user.last_name || ''}`.trim() : '';
      base.guardian_organization = instName;
      base.guardian_phone = user?.phone || meData?.contact_phone || '';
      base.guardian_email = user?.email || meData?.contact_email || '';
      base.guardian_address = meData?.address || '';
    }
    return base;
  }, [editingPatient, initialFormOverrides, isCuratelleInstitution, userIsCurator, meData]);

  const [formData, setFormData] = useState(buildInitialForm);
  const [avsRevealed, setAvsRevealed] = useState(false);
  const [duplicateWarning, setDuplicateWarning] = useState(null);

  const handleChange = (field, value) => setFormData(prev => ({ ...prev, [field]: value }));

  const maskAvs = useCallback((avs) => {
    if (!avs) return '';
    if (avs.length <= 4) return avs;
    return avs.substring(0, 4) + avs.substring(4).replace(/[0-9]/g, '*');
  }, []);

  const handleSubmit = async (e, forceCreate = false) => {
    e?.preventDefault?.();
    if (!formData.first_name || !formData.last_name) {
      toast.error('Prénom et nom requis');
      return;
    }
    const payload = { ...formData };
    Object.keys(payload).forEach(key => { if (payload[key] === '') payload[key] = null; });
    payload.first_name = formData.first_name;
    payload.last_name = formData.last_name;
    if (forceCreate) payload.force_create = true;

    try {
      if (editingPatient) {
        await updateMutation.mutateAsync({ patientId: editingPatient.id, data: payload });
        toast.success('Patient mis à jour');
        onSaved?.({ ...editingPatient, ...payload });
      } else {
        const result = await createMutation.mutateAsync(payload);
        const syncStatus = result?.sync?.status;
        const suggestionsCount = result?.sync?.suggestions_count || 0;

        if (syncStatus === 'linked') {
          toast.success('Patient lié automatiquement — données en cours de synchronisation');
        } else if (syncStatus === 'suggestions' && suggestionsCount > 0) {
          toast.info(`${suggestionsCount} correspondance${suggestionsCount > 1 ? 's' : ''} potentielle${suggestionsCount > 1 ? 's' : ''} trouvée${suggestionsCount > 1 ? 's' : ''} — vérification requise`);
        } else {
          toast.success('Patient créé');
        }
        onSaved?.(result?.patient || result);
      }
      onClose();
    } catch (err) {
      const data = err?.response?.data;
      if (data?.code === 'DUPLICATE_PATIENT' && data?.duplicates) {
        setDuplicateWarning(data.duplicates);
      } else {
        toast.error(data?.error || 'Erreur');
      }
    }
  };

  const filled = [formData.first_name, formData.last_name, formData.dob, formData.phone, formData.address, formData.city].filter(Boolean).length;
  const pct = Math.round((filled / 6) * 100);

  return (
    <div className={s.modal} onClick={onClose}>
      <div className={s.modalContent} onClick={(e) => e.stopPropagation()}>

        {/* ── Header ── */}
        <div className={s.mHeader}>
          <div className={s.mHeaderLeft}>
            <div className={s.mAvatar}>
              {getInitials(formData.first_name, formData.last_name)}
            </div>
            <div>
              <h3 className={s.mTitle}>
                {editingPatient ? 'Modifier le patient' : 'Nouveau patient'}
              </h3>
              <p className={s.mSubtitle}>
                {formData.first_name || formData.last_name
                  ? `${GENDER_SHORT[formData.gender] ? GENDER_SHORT[formData.gender] + ' ' : ''}${formData.first_name} ${formData.last_name}`.trim()
                  : 'Renseignez les informations ci-dessous'}
              </p>
            </div>
          </div>
          <button className={s.mClose} onClick={onClose} type="button" aria-label="Fermer">
            <FaTimes />
          </button>
        </div>

        {/* ── Progress bar ── */}
        <div className={s.progressBar}>
          <div className={s.progressFill} style={{ width: `${pct}%` }} />
        </div>

        {/* ── Form ── */}
        <form onSubmit={handleSubmit}>
          <div className={s.mBody}>

            {/* ═══ Section 1 : Identité ═══ */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <FaUser className={s.sectionIcon} />
                <span>Identité</span>
              </div>
              <div className={s.sectionBody}>
                <div className={s.row3}>
                  <div className={s.field}>
                    <label>Civilité</label>
                    <ChipDropdown value={formData.gender} options={GENDER_OPTIONS} onChange={(v) => handleChange('gender', v)} />
                  </div>
                  <div className={s.field}>
                    <label htmlFor="patient-first-name">Prénom <span className={s.req}>*</span></label>
                    <input id="patient-first-name" type="text" value={formData.first_name} onChange={(e) => handleChange('first_name', capitalizeFirstName(e.target.value))} required />
                  </div>
                  <div className={s.field}>
                    <label htmlFor="patient-last-name">Nom <span className={s.req}>*</span></label>
                    <input id="patient-last-name" type="text" value={formData.last_name} onChange={(e) => handleChange('last_name', e.target.value.toUpperCase())} required style={{ textTransform: 'uppercase' }} />
                  </div>
                </div>
                <div className={s.row}>
                  <div className={s.field}>
                    <label>Date de naissance</label>
                    <InlineDatePicker value={formData.dob} onChange={(iso) => handleChange('dob', iso)} placeholder="Naissance" />
                  </div>
                  <div className={s.field}>
                    <label>Téléphone</label>
                    <input type="tel" value={formData.phone} onChange={(e) => handleChange('phone', e.target.value)} placeholder="+41 79 123 45 67" />
                  </div>
                </div>
              </div>
            </div>

            {/* ═══ Section 2 : Adresse ═══ */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <FaMapMarkerAlt className={s.sectionIcon} />
                <span>Adresse du domicile</span>
              </div>
              <div className={s.sectionBody}>
                <div className={s.field}>
                  <label>Adresse</label>
                  <AddressAutocomplete
                    name="patient_address"
                    inputId="patient_address"
                    value={formData.address ? [formData.address, formData.postal_code, formData.city].filter(Boolean).join(', ') : ''}
                    onChange={(e) => handleChange('address', e.target.value)}
                    onSelect={(item) => {
                      setFormData((prev) => ({
                        ...prev,
                        address: item.address || item.label || '',
                        postal_code: item.postcode || '',
                        city: item.city || '',
                      }));
                    }}
                    placeholder="Tapez pour rechercher une adresse..."
                    inputClassName={s.addressInput}
                  />
                </div>
                <div className={s.row}>
                  <div className={s.field}>
                    <label>NPA</label>
                    <input type="text" value={formData.postal_code} onChange={(e) => handleChange('postal_code', e.target.value)} placeholder="1247" />
                  </div>
                  <div className={s.field}>
                    <label>Ville</label>
                    <input type="text" value={formData.city} onChange={(e) => handleChange('city', e.target.value)} placeholder="Anières" />
                  </div>
                </div>
              </div>
            </div>

            {/* ═══ Section 3 : Accès & logistique ═══ */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <FaTruck className={s.sectionIcon} />
                <span>Accès & logistique</span>
              </div>
              <p className={s.sectionHint}>Informations transmises au chauffeur lors de la prise en charge.</p>
              <div className={s.sectionBody}>
                <div className={s.row}>
                  <div className={s.field}>
                    <label>Code porte</label>
                    <input type="text" value={formData.door_code} onChange={(e) => handleChange('door_code', e.target.value)} placeholder="1234 ou A56B" />
                  </div>
                  <div className={s.field}>
                    <label>Étage</label>
                    <input type="text" value={formData.floor} onChange={(e) => handleChange('floor', e.target.value)} placeholder="3, RDC, 2B..." />
                  </div>
                </div>
                <div className={s.field}>
                  <label>Notes d'accès</label>
                  <textarea value={formData.access_notes} onChange={(e) => handleChange('access_notes', e.target.value)} rows={2} placeholder="Sonner chez le concierge, ascenseur petit..." />
                </div>
                <div className={s.field}>
                  <label>Établissement de résidence</label>
                  <input type="text" value={formData.residence_name} onChange={(e) => handleChange('residence_name', e.target.value)} placeholder="EMS Les Tilleuls, Foyer de la Rive..." />
                  <span className={s.fieldHint}>Si le patient réside dans un EMS, foyer ou autre établissement.</span>
                </div>
              </div>
            </div>

            {/* ═══ Section 4 : Informations administratives ═══ */}
            {canSeeAdmin && (
              <div className={`${s.section} ${s.sectionRestricted}`}>
                <div className={s.sectionHeader}>
                  <FaShieldAlt className={s.sectionIcon} />
                  <span>Informations administratives</span>
                  <span className={s.restrictedBadge}><FaLock size={9} /> Restreint</span>
                </div>
                <p className={s.sectionHint}>Visible uniquement par l'équipe administrative. Jamais transmis aux transporteurs.</p>
                <div className={s.sectionBody}>
                  <div className={s.row}>
                    <div className={s.field}>
                      <label>N° AVS</label>
                      {canEditAdmin ? (
                        <div className={s.avsField}>
                          <input type={avsRevealed ? 'text' : 'password'} value={formData.avs_number} onChange={(e) => handleChange('avs_number', e.target.value)} placeholder="756.XXXX.XXXX.XX" autoComplete="off" />
                          <button type="button" className={s.avsToggle} onClick={() => setAvsRevealed(prev => !prev)} title={avsRevealed ? 'Masquer' : 'Afficher'}>
                            {avsRevealed ? <FaEyeSlash size={13} /> : <FaEye size={13} />}
                          </button>
                        </div>
                      ) : (
                        <span className={s.maskedValue}>{maskAvs(formData.avs_number) || '—'}</span>
                      )}
                      <span className={s.fieldHint}>Donnée sensible — accès restreint.</span>
                    </div>
                    <div className={s.field}>
                      <label>N° assuré</label>
                      <input type="text" value={formData.insurance_number} onChange={(e) => handleChange('insurance_number', e.target.value)} placeholder="Numéro de police" disabled={!canEditAdmin} />
                    </div>
                  </div>
                  <div className={s.field}>
                    <label>Caisse maladie</label>
                    <input type="text" value={formData.insurance_name} onChange={(e) => handleChange('insurance_name', e.target.value)} placeholder="CSS, Helsana, Assura..." disabled={!canEditAdmin} />
                  </div>
                </div>
              </div>
            )}

            {/* ═══ Section 4b : Curatelle ═══ */}
            {canSeeAdmin && (
              <div className={`${s.section} ${formData.has_guardianship ? s.sectionCuratelle : ''}`}>
                <div className={s.sectionHeader}>
                  <FaGavel className={s.sectionIcon} />
                  <span>Curatelle / Représentation légale</span>
                  {formData.has_guardianship && formData.guardianship_type && (
                    <span
                      className={s.guardianshipTypeBadge}
                      style={{
                        background: (GUARDIANSHIP_TYPE_COLORS[formData.guardianship_type] || {}).bg,
                        color: (GUARDIANSHIP_TYPE_COLORS[formData.guardianship_type] || {}).text,
                        borderColor: (GUARDIANSHIP_TYPE_COLORS[formData.guardianship_type] || {}).border,
                      }}
                    >
                      {GUARDIANSHIP_TYPE_LABELS[formData.guardianship_type] || formData.guardianship_type}
                    </span>
                  )}
                </div>
                <div className={s.sectionBody}>
                  <div className={s.field}>
                    <div className={s.checkboxField}>
                      <input type="checkbox" id="has_guardianship" checked={formData.has_guardianship} onChange={(e) => handleChange('has_guardianship', e.target.checked)} disabled={!canEditAdmin} />
                      <label htmlFor="has_guardianship">Patient sous curatelle ou représentation légale</label>
                    </div>
                  </div>
                  {formData.has_guardianship && (
                    <>
                      <div className={s.field}>
                        <label>Type de mesure <span className={s.req}>*</span></label>
                        <ChipDropdown
                          value={formData.guardianship_type}
                          options={GUARDIANSHIP_TYPES}
                          onChange={(v) => handleChange('guardianship_type', v)}
                          disabled={!canEditAdmin}
                        />
                        <span className={s.fieldHint}>
                          {formData.guardianship_type === 'opad' && 'Service public de protection de l\'adulte — facturation gérée automatiquement.'}
                          {formData.guardianship_type === 'lawyer' && 'Avocat ou étude juridique mandaté(e) — adresse utilisée pour la facturation.'}
                          {formData.guardianship_type === 'curatorship' && 'Curateur professionnel désigné — l\'adresse de facturation sera celle du curateur.'}
                          {formData.guardianship_type === 'family' && 'Membre de la famille désigné comme représentant légal.'}
                          {formData.guardianship_type === 'other' && 'Autre type de représentation ou mesure de protection.'}
                        </span>
                      </div>

                      <div className={s.row}>
                        <div className={s.field}>
                          <label>Nom du curateur / représentant</label>
                          <input type="text" value={formData.guardian_name} onChange={(e) => handleChange('guardian_name', e.target.value)} placeholder="Me. Dupont, M. Martin..." disabled={!canEditAdmin} />
                        </div>
                        <div className={s.field}>
                          <label>Organisation</label>
                          <input type="text" value={formData.guardian_organization} onChange={(e) => handleChange('guardian_organization', e.target.value)}
                            placeholder={
                              formData.guardianship_type === 'opad' ? 'OPAD Genève, SPAd Vaud...'
                              : formData.guardianship_type === 'lawyer' ? 'Étude Me. Dupont & Associés...'
                              : 'Nom de l\'organisme...'
                            }
                            disabled={!canEditAdmin}
                          />
                        </div>
                      </div>

                      <div className={s.row}>
                        <div className={s.field}>
                          <label><FaPhone size={9} style={{marginRight: 4, opacity: 0.5}} />Téléphone</label>
                          <input type="tel" value={formData.guardian_phone} onChange={(e) => handleChange('guardian_phone', e.target.value)} placeholder="+41 22 000 00 00" disabled={!canEditAdmin} />
                        </div>
                        <div className={s.field}>
                          <label><FaEnvelope size={9} style={{marginRight: 4, opacity: 0.5}} />Email</label>
                          <input type="email" value={formData.guardian_email} onChange={(e) => handleChange('guardian_email', e.target.value)} placeholder="curateur@example.ch" disabled={!canEditAdmin} />
                        </div>
                      </div>

                      <div className={s.field}>
                        <label><FaMapMarkerAlt size={9} style={{marginRight: 4, opacity: 0.5}} />Adresse du curateur</label>
                        {canEditAdmin ? (
                          <AddressAutocomplete
                            name="guardian_address"
                            inputId="guardian_address"
                            value={formData.guardian_address || ''}
                            onChange={(e) => handleChange('guardian_address', e.target.value)}
                            onSelect={(item) => {
                              const parts = [
                                item.address || item.label || '',
                                item.postcode || '',
                                item.city || '',
                              ].filter(Boolean);
                              handleChange('guardian_address', parts.join(', '));
                            }}
                            placeholder="Tapez pour rechercher une adresse..."
                            inputClassName={s.addressInput}
                          />
                        ) : (
                          <div className={s.maskedValue}>{formData.guardian_address || '—'}</div>
                        )}
                        <span className={s.fieldHint}>
                          Adresse utilisée automatiquement pour la facturation lorsque le patient est facturé via son curateur.
                        </span>
                      </div>

                      <div className={s.billingAutoInfo}>
                        <FaShieldAlt size={12} />
                        <div>
                          <strong>Facturation automatique</strong>
                          <p>Lorsqu'un transport est facturé au curateur, le système utilisera automatiquement le nom, l'adresse et les coordonnées renseignés ci-dessus pour créer le tiers payeur.</p>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* ═══ Section 5 : Notes ═══ */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <FaStickyNote className={s.sectionIcon} />
                <span>Notes internes</span>
              </div>
              <div className={s.sectionBody}>
                <div className={s.field}>
                  <textarea value={formData.notes} onChange={(e) => handleChange('notes', e.target.value)} rows={3} placeholder="Notes visibles uniquement par votre équipe..." />
                  <span className={s.fieldHint}>Ces informations ne sont jamais transmises aux transporteurs.</span>
                </div>
              </div>
            </div>

          </div>

          {/* ── Avertissement doublon ── */}
          {duplicateWarning && (
            <div style={{
              margin: '0 16px 12px',
              padding: '14px 16px',
              background: '#FFF8E1',
              border: '1px solid #FFD54F',
              borderRadius: 8,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <FaShieldAlt style={{ color: '#F57F17' }} />
                <strong style={{ fontSize: 13, color: '#E65100' }}>
                  Patient potentiellement en doublon
                </strong>
              </div>
              <p style={{ fontSize: 12, color: '#5D4037', margin: '0 0 10px', lineHeight: 1.5 }}>
                Un ou plusieurs patients avec le même nom existent déjà dans votre institution :
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
                {duplicateWarning.map((dup) => (
                  <div key={dup.id} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '8px 12px',
                    background: '#fff',
                    borderRadius: 6,
                    border: '1px solid #FFE082',
                    fontSize: 12,
                  }}>
                    <strong style={{ color: '#333' }}>{dup.last_name} {dup.first_name}</strong>
                    {dup.dob && <span style={{ color: '#888' }}>{fmtDate(dup.dob)}</span>}
                    {dup.phone && <span style={{ color: '#888' }}>{dup.phone}</span>}
                    {dup.address && <span style={{ color: '#aaa', fontSize: 11 }}>{dup.address}</span>}
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  type="button"
                  onClick={() => setDuplicateWarning(null)}
                  style={{
                    padding: '6px 14px',
                    borderRadius: 6,
                    border: '1px solid #ccc',
                    background: '#fff',
                    fontSize: 12,
                    cursor: 'pointer',
                  }}
                >
                  Annuler
                </button>
                <button
                  type="button"
                  onClick={(e) => handleSubmit(e, true)}
                  style={{
                    padding: '6px 14px',
                    borderRadius: 6,
                    border: 'none',
                    background: '#E65100',
                    color: '#fff',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  Créer quand même
                </button>
              </div>
            </div>
          )}

          {/* ── Footer ── */}
          <div className={s.mFooter}>
            <div className={s.mFooterInfo}>
              {formData.first_name || formData.last_name ? (
                <span className={s.mFooterName}>
                  {formData.first_name} {formData.last_name}
                  {formData.dob && <span className={s.mFooterDob}> — {fmtDate(formData.dob)}</span>}
                </span>
              ) : (
                <span className={s.mFooterEmpty}>Aucun patient renseigné</span>
              )}
            </div>
            <div className={s.mFooterActions}>
              <button type="button" className={s.mCancelBtn} onClick={onClose}>
                Annuler
              </button>
              <button type="submit" className={s.mSubmitBtn} disabled={createMutation.isPending || updateMutation.isPending}>
                <FaCheck size={11} />
                {editingPatient ? 'Enregistrer' : 'Créer le patient'}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
