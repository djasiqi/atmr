// src/pages/company/components/EditDriverForm.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { FiEdit2, FiX, FiUser, FiTruck, FiShield, FiLock, FiAlertTriangle, FiChevronDown, FiCopy, FiCheck } from 'react-icons/fi';
import { toast } from 'sonner';
import s from './EditDriverForm.module.css';
import ntStyles from '../Settings/tabs/NotificationsTab.module.css';
import { fetchCompanyVehicles } from '../../../services/companyService';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import apiClient from '../../../utils/apiClient';

const formatDate = (iso) => {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit', year: 'numeric' });
  } catch {
    return iso;
  }
};

const PLACEHOLDER_VALUES = ['non spécifié', 'non specifie', 'non renseigné', 'non renseigne', '—', '-'];

const clean = (val) => {
  if (!val) return '';
  const str = String(val).trim();
  if (PLACEHOLDER_VALUES.includes(str.toLowerCase())) return '';
  return str;
};

const CONTRACT_OPTIONS = [
  { value: 'CDI', label: 'CDI' },
  { value: 'CDD', label: 'CDD' },
  { value: 'INTERIM', label: 'Intérimaire' },
  { value: 'INDEPENDANT', label: 'Indépendant' },
];

function ContractChipDropdown({ value, onChange, disabled }) {
  const [open, setOpen] = React.useState(false);
  const btnRef = React.useRef(null);
  const menuRef = React.useRef(null);
  const [pos, setPos] = React.useState({ top: 0, left: 0, width: 0 });

  React.useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const reposition = React.useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, left: r.left, width: r.width });
  }, []);

  React.useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => { window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition); };
  }, [open, reposition]);

  const selected = CONTRACT_OPTIONS.find((o) => o.value === value) || CONTRACT_OPTIONS[0];

  return (
    <div className={s.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${s.chipBtn} ${value ? s.chipBtnActive : ''}`}
        onClick={() => !disabled && setOpen((p) => !p)}
        disabled={disabled}
      >
        <span className={s.chipText}>{selected.label}</span>
        <FiChevronDown size={11} className={`${s.chipArrow} ${open ? s.chipArrowOpen : ''}`} />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className={s.chipMenu}
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: pos.width, zIndex: 10000 }}
        >
          {CONTRACT_OPTIONS.map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${s.chipOption} ${o.value === value ? s.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value); setOpen(false); }}
            >
              {o.label}
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}

function VehicleChipDropdown({ vehicles, driverId, value, onChange, disabled }) {
  const [open, setOpen] = React.useState(false);
  const btnRef = React.useRef(null);
  const menuRef = React.useRef(null);
  const [pos, setPos] = React.useState({ top: 0, left: 0, width: 0 });

  React.useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const reposition = React.useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, left: r.left, width: Math.max(r.width, 280) });
  }, []);

  React.useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => { window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition); };
  }, [open, reposition]);

  const options = React.useMemo(() => {
    const list = [{ value: '', label: 'Aucun', suffix: '' }];
    vehicles.forEach((v) => {
      const isAssignedToOther = v.assigned_driver_id && v.assigned_driver_id !== driverId;
      const label = `${v.model} - ${v.license_plate}${v.year ? ` (${v.year})` : ''}`;
      const suffix = isAssignedToOther
        ? ` [${v.assigned_driver_name || 'Assigné'}]`
        : v.assigned_driver_id === driverId ? ' [Actuel]' : '';
      list.push({ value: String(v.id), label, suffix });
    });
    return list;
  }, [vehicles, driverId]);

  const selected = options.find((o) => o.value === String(value || '')) || options[0];

  return (
    <div className={s.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${s.chipBtn} ${value ? s.chipBtnActive : ''}`}
        onClick={() => !disabled && setOpen((p) => !p)}
        disabled={disabled}
      >
        <span className={s.chipText}>{selected.label}{selected.suffix}</span>
        <FiChevronDown size={11} className={`${s.chipArrow} ${open ? s.chipArrowOpen : ''}`} />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className={s.chipMenu}
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: pos.width, zIndex: 10000 }}
        >
          {options.map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${s.chipOption} ${o.value === String(value || '') ? s.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value || ''); setOpen(false); }}
            >
              {o.label}<span style={{ opacity: 0.5, fontSize: 11 }}>{o.suffix}</span>
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}

const EditDriverForm = ({ driver, onSubmit, onClose }) => {
  const [editing, setEditing] = useState(false);

  // User fields — clean placeholders from backend
  const [userData, setUserData] = useState({
    first_name: clean(driver.first_name) || clean(driver.user?.first_name),
    last_name: clean(driver.last_name) || clean(driver.user?.last_name),
    email: clean(driver.email) || clean(driver.user?.email),
    phone: clean(driver.phone) || clean(driver.user?.phone),
    birth_date: clean(driver.birth_date) || clean(driver.user?.birth_date),
  });

  // Driver fields
  const [formData, setFormData] = useState({
    vehicle_id: driver.vehicle_id || driver.vehicle?.id || null,
    is_active: driver.is_active !== undefined ? driver.is_active : true,
    avs_number: clean(driver.avs_number),
    nationality: clean(driver.nationality),
    contract_type: driver.contract_type || 'CDI',
    emergency_contact_name: clean(driver.emergency_contact_name),
    emergency_contact_phone: clean(driver.emergency_contact_phone),
    license_categories: driver.license_categories || [],
    license_valid_until: clean(driver.license_valid_until),
    medical_valid_until: clean(driver.medical_valid_until),
    employment_start_date: clean(driver.employment_start_date),
  });

  const [domicileAddress, setDomicileAddress] = useState(
    clean(driver.address) || clean(driver.user?.address)
  );
  const [vehicles, setVehicles] = useState([]);
  const [loadingVehicles, setLoadingVehicles] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isResettingPassword, setIsResettingPassword] = useState(false);
  const [resetStep, setResetStep] = useState('idle'); // 'idle' | 'confirm' | 'done'
  const [generatedPassword, setGeneratedPassword] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const loadVehicles = async () => {
      try {
        setLoadingVehicles(true);
        const list = await fetchCompanyVehicles();
        setVehicles((list || []).filter((v) => v.is_active !== false));
      } catch {
        toast.error('Impossible de charger les vehicules');
      } finally {
        setLoadingVehicles(false);
      }
    };
    loadVehicles();
  }, []);

  const handleUserChange = (e) => {
    const { name, value } = e.target;
    setUserData((prev) => ({ ...prev, [name]: value }));
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value === '' ? null : value,
    }));
  };

  const handleDomicileAddressChange = (e) => {
    let addr = '';
    if (e && typeof e === 'object' && e.target) addr = e.target.value || '';
    else if (typeof e === 'string') addr = e;
    setDomicileAddress(String(addr || '').trim());
  };

  const handleDomicileAddressSelect = (item) => {
    let addr = '';
    if (item && typeof item === 'object') addr = item.label || item.address || '';
    else if (typeof item === 'string') addr = item;
    const clean = String(addr || '').trim();
    if (clean) setDomicileAddress(clean);
  };

  const handleResetPassword = useCallback(async () => {
    setIsResettingPassword(true);
    try {
      const res = await apiClient.post(`/companies/me/drivers/${driver.id}/reset-password`);
      if (res.data?.new_password) {
        setGeneratedPassword(res.data.new_password);
        setResetStep('done');
      } else {
        toast.error('Erreur lors de la réinitialisation.');
        setResetStep('idle');
      }
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la réinitialisation.');
      setResetStep('idle');
    } finally {
      setIsResettingPassword(false);
    }
  }, [driver.id]);

  const handleCopyPassword = useCallback(() => {
    navigator.clipboard.writeText(generatedPassword).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [generatedPassword]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      const payload = {
        first_name: userData.first_name.trim(),
        last_name: userData.last_name.trim(),
        email: userData.email.trim(),
        phone: userData.phone.trim(),
        birth_date: userData.birth_date || null,
        address: domicileAddress.trim(),
        vehicle_id: formData.vehicle_id ? Number(formData.vehicle_id) : null,
        is_active: formData.is_active,
        avs_number: formData.avs_number || null,
        nationality: formData.nationality || null,
        contract_type: formData.contract_type || 'CDI',
        emergency_contact_name: formData.emergency_contact_name || null,
        emergency_contact_phone: formData.emergency_contact_phone || null,
        license_categories: formData.license_categories || [],
        license_valid_until: formData.license_valid_until || null,
        medical_valid_until: formData.medical_valid_until || null,
        employment_start_date: formData.employment_start_date || null,
      };
      await onSubmit(driver.id, payload);
      setEditing(false);
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la mise a jour.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const selectedVehicle = vehicles.find((v) => v.id === Number(formData.vehicle_id));
  const vehicleLabel = selectedVehicle
    ? `${selectedVehicle.model} - ${selectedVehicle.license_plate}`
    : clean(driver.vehicle_assigned) || null;
  const fullName = `${userData.first_name} ${userData.last_name}`.trim() || driver.username;

  // Helper: only render a field if it has a value
  const Field = ({ label, value, full }) => {
    if (!value) return null;
    return (
      <div className={`${s.readField}${full ? ` ${s.readFieldFull}` : ''}`}>
        <span className={s.readLabel}>{label}</span>
        <span className={s.readValue}>{value}</span>
      </div>
    );
  };

  // Pre-compute optional values
  const birthDateFmt = formatDate(userData.birth_date);
  const licenseCategories = formData.license_categories && formData.license_categories.length > 0
    ? formData.license_categories.join(', ')
    : null;
  const licenseValidFmt = formatDate(formData.license_valid_until);
  const medicalValidFmt = formatDate(formData.medical_valid_until);
  const employmentStartFmt = formatDate(formData.employment_start_date);

  const hasVehicle = !!vehicleLabel;
  const hasPermis = !!(licenseCategories || licenseValidFmt || medicalValidFmt || employmentStartFmt);
  const hasEmergency = !!(formData.emergency_contact_name || formData.emergency_contact_phone);

  // ====================== READ MODE ======================
  if (!editing) {
    return (
      <div className={s.readPanel}>
        {/* Header */}
        <div className={s.readHeader}>
          <div className={s.readHeaderRow}>
            <h3 className={s.readTitle}>{fullName}</h3>
            <button type="button" className={s.closeBtn} onClick={onClose}>
              <FiX size={18} />
            </button>
          </div>
          <div className={s.readMeta}>
            <span className={`${s.statusBadge} ${formData.is_active ? s.statusActive : s.statusInactive}`}>
              {formData.is_active ? 'Actif' : 'Inactif'}
            </span>
            {formData.contract_type && (
              <span className={s.contractBadge}>{CONTRACT_OPTIONS.find((o) => o.value === formData.contract_type)?.label || formData.contract_type}</span>
            )}
          </div>
        </div>

        {/* Identity — always shown (name + email are required) */}
        <div className={s.readSection}>
          <div className={s.readSectionHeader}>
            <FiUser size={14} />
            <span>Identite</span>
          </div>
          <div className={s.readGrid}>
            <Field label="Prenom" value={userData.first_name} />
            <Field label="Nom" value={userData.last_name} />
            <Field label="Date de naissance" value={birthDateFmt} />
            <Field label="Nationalite" value={formData.nationality} />
            <Field label="N° AVS" value={formData.avs_number} full />
            <Field label="Email" value={userData.email} full />
            <Field label="Telephone" value={userData.phone} full />
            <Field label="Adresse" value={domicileAddress} full />
          </div>
        </div>

        {/* Vehicle — only if assigned */}
        {hasVehicle && (
          <div className={s.readSection}>
            <div className={s.readSectionHeader}>
              <FiTruck size={14} />
              <span>Vehicule</span>
            </div>
            <div className={s.readGrid}>
              <Field label="Vehicule assigne" value={vehicleLabel} full />
            </div>
          </div>
        )}

        {/* Permis & Medical — only if at least one field filled */}
        {hasPermis && (
          <div className={s.readSection}>
            <div className={s.readSectionHeader}>
              <FiShield size={14} />
              <span>Permis et medical</span>
            </div>
            <div className={s.readGrid}>
              <Field label="Categories" value={licenseCategories} />
              <Field label="Validite permis" value={licenseValidFmt} />
              <Field label="Validite medical" value={medicalValidFmt} />
              <Field label="Debut emploi" value={employmentStartFmt} />
            </div>
          </div>
        )}

        {/* Emergency — only if at least one field filled */}
        {hasEmergency && (
          <div className={s.readSection}>
            <div className={s.readSectionHeader}>
              <FiAlertTriangle size={14} />
              <span>Contact d&apos;urgence</span>
            </div>
            <div className={s.readGrid}>
              <Field label="Nom" value={formData.emergency_contact_name} />
              <Field label="Telephone" value={formData.emergency_contact_phone} />
            </div>
          </div>
        )}

        {/* Reset password inline flow */}
        {resetStep === 'confirm' && (
          <div className={s.resetBanner}>
            <div className={s.resetBannerIcon}><FiLock size={14} /></div>
            <div className={s.resetBannerBody}>
              <strong>Réinitialiser le mot de passe ?</strong>
              <span>Un nouveau mot de passe sera généré pour ce chauffeur.</span>
            </div>
            <div className={s.resetBannerActions}>
              <button type="button" className={s.resetBannerCancel} onClick={() => setResetStep('idle')}>Annuler</button>
              <button type="button" className={s.resetBannerConfirm} onClick={handleResetPassword} disabled={isResettingPassword}>
                {isResettingPassword ? 'Génération...' : 'Confirmer'}
              </button>
            </div>
          </div>
        )}

        {resetStep === 'done' && (
          <div className={s.resetResult}>
            <div className={s.resetResultIcon}><FiCheck size={14} /></div>
            <div className={s.resetResultBody}>
              <strong>Nouveau mot de passe généré</strong>
              <span>Communiquez-le au chauffeur de manière sécurisée.</span>
            </div>
            <div className={s.resetResultPassword}>
              <code className={s.resetResultCode}>{generatedPassword}</code>
              <button type="button" className={s.resetResultCopy} onClick={handleCopyPassword} title="Copier">
                {copied ? <FiCheck size={13} /> : <FiCopy size={13} />}
              </button>
            </div>
            <button type="button" className={s.resetResultDismiss} onClick={() => { setResetStep('idle'); setGeneratedPassword(''); setCopied(false); }}>
              Fermer
            </button>
          </div>
        )}

        {/* Actions */}
        <div className={s.readActions}>
          <button
            type="button"
            className={s.resetPasswordButton}
            onClick={() => setResetStep('confirm')}
            disabled={resetStep !== 'idle'}
          >
            <FiLock size={12} />
            Réinitialiser le mot de passe
          </button>
          <button type="button" className={s.editBtn} onClick={() => setEditing(true)}>
            <FiEdit2 size={14} />
            Modifier
          </button>
        </div>
      </div>
    );
  }

  // ====================== EDIT MODE ======================
  return (
    <form onSubmit={handleSubmit} className={s.form}>
      {/* Header */}
      <div className={s.editHeader}>
        <h3 className={s.editTitle}>Modifier {fullName}</h3>
        <button type="button" className={s.closeBtn} onClick={() => setEditing(false)}>
          <FiX size={18} />
        </button>
      </div>

      {/* Identity */}
      <div className={s.section}>
        <div className={s.sectionHeader}><FiUser size={13} /><span>Identite</span></div>
        <div className={s.formRow}>
          <div className={s.formGroup}>
            <label htmlFor="first_name">Prenom <span className={s.required}>*</span></label>
            <input type="text" id="first_name" name="first_name" value={userData.first_name} onChange={handleUserChange} placeholder="Prenom" required disabled={isSubmitting} />
          </div>
          <div className={s.formGroup}>
            <label htmlFor="last_name">Nom <span className={s.required}>*</span></label>
            <input type="text" id="last_name" name="last_name" value={userData.last_name} onChange={handleUserChange} placeholder="Nom" required disabled={isSubmitting} />
          </div>
        </div>
        <div className={s.formRow}>
          <div className={s.formGroup}>
            <label>Date de naissance</label>
            <InlineDatePicker value={userData.birth_date || ''} onChange={(v) => handleUserChange({ target: { name: 'birth_date', value: v } })} placeholder="Naissance" />
          </div>
          <div className={s.formGroup}>
            <label htmlFor="nationality">Nationalite</label>
            <input type="text" id="nationality" name="nationality" value={formData.nationality || ''} onChange={handleChange} placeholder="ex: Suisse" disabled={isSubmitting} />
          </div>
        </div>
        <div className={s.formGroup}>
          <label htmlFor="avs_number">N° AVS</label>
          <input type="text" id="avs_number" name="avs_number" value={formData.avs_number || ''} onChange={handleChange} placeholder="756.XXXX.XXXX.XX" disabled={isSubmitting} />
        </div>
        <div className={s.formGroup}>
          <label htmlFor="email">Email <span className={s.required}>*</span></label>
          <input type="email" id="email" name="email" value={userData.email} onChange={handleUserChange} placeholder="email@exemple.com" required disabled={isSubmitting} />
        </div>
        <div className={s.formGroup}>
          <label htmlFor="phone">Telephone</label>
          <input type="tel" id="phone" name="phone" value={userData.phone || ''} onChange={handleUserChange} placeholder="+41 7X XXX XX XX" disabled={isSubmitting} />
        </div>
        <div className={s.formGroup}>
          <label htmlFor="domicile_address">Adresse</label>
          <AddressAutocomplete id="domicile_address" name="domicile_address" value={domicileAddress} onChange={handleDomicileAddressChange} onSelect={handleDomicileAddressSelect} placeholder="Adresse complete" disabled={isSubmitting} />
        </div>
      </div>

      {/* Vehicle */}
      <div className={s.section}>
        <div className={s.sectionHeader}><FiTruck size={13} /><span>Vehicule</span></div>
        <div className={s.formGroup}>
          <label>Vehicule assigne</label>
          {loadingVehicles ? (
            <div className={s.loadingText}>Chargement...</div>
          ) : vehicles.length === 0 ? (
            <div className={s.warningText}>Aucun vehicule disponible.</div>
          ) : (
            <VehicleChipDropdown
              vehicles={vehicles}
              driverId={driver.id}
              value={formData.vehicle_id || ''}
              onChange={(v) => handleChange({ target: { name: 'vehicle_id', value: v } })}
              disabled={isSubmitting}
            />
          )}
        </div>
      </div>

      {/* Permis & Contrat */}
      <div className={s.section}>
        <div className={s.sectionHeader}><FiShield size={13} /><span>Permis et contrat</span></div>
        <div className={s.formRow}>
          <div className={s.formGroup}>
            <label>Contrat</label>
            <ContractChipDropdown
              value={formData.contract_type || 'CDI'}
              onChange={(v) => handleChange({ target: { name: 'contract_type', value: v } })}
              disabled={isSubmitting}
            />
          </div>
          <div className={s.formGroup}>
            <label>Debut emploi</label>
            <InlineDatePicker value={formData.employment_start_date || ''} onChange={(v) => handleChange({ target: { name: 'employment_start_date', value: v } })} placeholder="Début" />
          </div>
        </div>
        <div className={s.formRow}>
          <div className={s.formGroup}>
            <label>Validite permis</label>
            <InlineDatePicker value={formData.license_valid_until || ''} onChange={(v) => handleChange({ target: { name: 'license_valid_until', value: v } })} placeholder="Permis" />
          </div>
          <div className={s.formGroup}>
            <label>Validite medical</label>
            <InlineDatePicker value={formData.medical_valid_until || ''} onChange={(v) => handleChange({ target: { name: 'medical_valid_until', value: v } })} placeholder="Médical" />
          </div>
        </div>
      </div>

      {/* Emergency */}
      <div className={s.section}>
        <div className={s.sectionHeader}><FiAlertTriangle size={13} /><span>Contact d&apos;urgence</span></div>
        <div className={s.formRow}>
          <div className={s.formGroup}>
            <label htmlFor="emergency_contact_name">Nom</label>
            <input type="text" id="emergency_contact_name" name="emergency_contact_name" value={formData.emergency_contact_name || ''} onChange={handleChange} placeholder="Nom du contact" disabled={isSubmitting} />
          </div>
          <div className={s.formGroup}>
            <label htmlFor="emergency_contact_phone">Telephone</label>
            <input type="tel" id="emergency_contact_phone" name="emergency_contact_phone" value={formData.emergency_contact_phone || ''} onChange={handleChange} placeholder="+41 XX XXX XX XX" disabled={isSubmitting} />
          </div>
        </div>
      </div>

      {/* Status */}
      <div className={s.section}>
        <div className={s.sectionHeader}><FiUser size={13} /><span>Statut</span></div>
        <label className={`${ntStyles.notifRow} ${s.toggleRow}`} htmlFor="toggle-is-active">
          <div className={ntStyles.notifInfo}>
            <span className={ntStyles.notifLabel}>Chauffeur actif</span>
            <span className={ntStyles.notifHint}>Les chauffeurs inactifs n'apparaissent pas dans les sélections</span>
          </div>
          <div className={ntStyles.miniToggle}>
            <input id="toggle-is-active" type="checkbox" name="is_active" checked={formData.is_active} onChange={handleChange} disabled={isSubmitting} />
            <span className={ntStyles.miniSlider} />
          </div>
        </label>
      </div>

      {/* Actions */}
      <div className={s.formActions}>
        <div className={s.buttonGroup}>
          <button type="button" onClick={() => setEditing(false)} className={s.cancelButton} disabled={isSubmitting}>
            Annuler
          </button>
          <button type="submit" className={s.submitButton} disabled={isSubmitting}>
            {isSubmitting ? 'Enregistrement...' : 'Enregistrer'}
          </button>
        </div>
      </div>
    </form>
  );
};

export default EditDriverForm;
