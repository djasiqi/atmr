// src/pages/company/components/AddDriverForm.jsx
import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
  FiX, FiUser, FiLock, FiTruck, FiEye, FiEyeOff,
  FiUserPlus, FiShield, FiAlertTriangle, FiChevronDown,
} from 'react-icons/fi';
import { toast } from 'sonner';
import s from './AddDriverForm.module.css';
import ntStyles from '../Settings/tabs/NotificationsTab.module.css';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import { fetchCompanyVehicles } from '../../../services/companyService';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';

const CONTRACT_OPTIONS = [
  { value: 'CDI', label: 'CDI' },
  { value: 'CDD', label: 'CDD' },
  { value: 'INTERIM', label: 'Intérimaire' },
  { value: 'INDEPENDANT', label: 'Indépendant' },
];

function PortalChipDropdown({ options, value, onChange, disabled, placeholder }) {
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
    setPos({ top: r.bottom + 4, left: r.left, width: Math.max(r.width, 220) });
  }, []);

  React.useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => { window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition); };
  }, [open, reposition]);

  const selected = options.find((o) => o.value === String(value || ''));

  return (
    <div className={s.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${s.chipBtn} ${value ? s.chipBtnActive : ''}`}
        onClick={() => !disabled && setOpen((p) => !p)}
        disabled={disabled}
      >
        <span className={s.chipText}>{selected?.label || placeholder || 'Sélectionner'}</span>
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

const AddDriverForm = ({ onSubmit, onClose }) => {
  const [formData, setFormData] = useState({
    username: '',
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    birthDate: '',
    nationality: '',
    avsNumber: '',
    password: '',
    confirmPassword: '',
    vehicleId: '',
    contractType: 'CDI',
    employmentStartDate: '',
    licenseValidUntil: '',
    medicalValidUntil: '',
    emergencyContactName: '',
    emergencyContactPhone: '',
    isActive: true,
  });

  const [domicileAddress, setDomicileAddress] = useState('');
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [vehicles, setVehicles] = useState([]);
  const [loadingVehicles, setLoadingVehicles] = useState(true);

  useEffect(() => {
    const load = async () => {
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
    load();
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
    if (errors[name]) setErrors((prev) => ({ ...prev, [name]: null }));
  };

  const handleAddressChange = (e) => {
    let addr = '';
    if (e && typeof e === 'object' && e.target) addr = e.target.value || '';
    else if (typeof e === 'string') addr = e;
    setDomicileAddress(String(addr || '').trim());
  };

  const handleAddressSelect = (item) => {
    let addr = '';
    if (item && typeof item === 'object') addr = item.label || item.address || '';
    else if (typeof item === 'string') addr = item;
    const clean = String(addr || '').trim();
    if (clean) setDomicileAddress(clean);
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.username.trim()) newErrors.username = "Nom d'utilisateur requis";
    if (!formData.firstName.trim()) newErrors.firstName = 'Prenom requis';
    if (!formData.lastName.trim()) newErrors.lastName = 'Nom requis';
    if (!formData.email.trim()) newErrors.email = 'Email requis';
    else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Email invalide';
    if (formData.password.length < 8) newErrors.password = 'Min. 8 caracteres';
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Les mots de passe ne correspondent pas';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;
    setIsSubmitting(true);

    const selectedVehicle = vehicles.find((v) => String(v.id) === String(formData.vehicleId));

    const payload = {
      username: formData.username.trim(),
      first_name: formData.firstName.trim(),
      last_name: formData.lastName.trim(),
      email: formData.email.trim(),
      phone: formData.phone.trim() || null,
      birth_date: formData.birthDate || null,
      nationality: formData.nationality.trim() || null,
      avs_number: formData.avsNumber.trim() || null,
      address: domicileAddress.trim() || null,
      password: formData.password,
      vehicle_id: formData.vehicleId ? Number(formData.vehicleId) : null,
      vehicle_assigned: selectedVehicle ? selectedVehicle.model : null,
      brand: selectedVehicle ? selectedVehicle.brand : null,
      license_plate: selectedVehicle ? selectedVehicle.license_plate : null,
      contract_type: formData.contractType || 'CDI',
      employment_start_date: formData.employmentStartDate || null,
      license_valid_until: formData.licenseValidUntil || null,
      medical_valid_until: formData.medicalValidUntil || null,
      emergency_contact_name: formData.emergencyContactName.trim() || null,
      emergency_contact_phone: formData.emergencyContactPhone.trim() || null,
      is_active: formData.isActive,
    };

    try {
      await onSubmit(payload);
    } catch (error) {
      console.error('Submission failed:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const passwordStrength = (() => {
    const p = formData.password;
    if (!p) return 0;
    let score = 0;
    if (p.length >= 8) score += 1;
    if (p.length >= 12) score += 1;
    if (/[A-Z]/.test(p) && /[a-z]/.test(p)) score += 1;
    if (/\d/.test(p)) score += 1;
    if (/[^A-Za-z0-9]/.test(p)) score += 1;
    return Math.min(score, 4);
  })();

  const strengthLabel = ['', 'Faible', 'Moyen', 'Bon', 'Fort'][passwordStrength] || '';
  const strengthColor = ['', '#ef4444', '#f59e0b', '#22c55e', '#059669'][passwordStrength] || '';

  return (
    <div className={s.panel}>
      {/* Header */}
      <div className={s.header}>
        <div className={s.headerLeft}>
          <div className={s.headerIcon}>
            <FiUserPlus size={16} />
          </div>
          <h3 className={s.headerTitle}>Nouveau chauffeur</h3>
        </div>
        <button type="button" className={s.closeBtn} onClick={onClose} aria-label="Fermer">
          <FiX size={18} />
        </button>
      </div>

      {/* Body — scrollable */}
      <form onSubmit={handleSubmit} className={s.body}>

        {/* ── Identite ── */}
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <FiUser size={13} />
            <span>Identite</span>
          </div>

          <div className={s.formGroup}>
            <label htmlFor="add_username">Identifiant <span className={s.required}>*</span></label>
            <input type="text" id="add_username" name="username" value={formData.username} onChange={handleChange} placeholder="jean.dupont" required disabled={isSubmitting} className={errors.username ? s.inputError : ''} />
            {errors.username && <span className={s.error}>{errors.username}</span>}
            <span className={s.hint}>Servira de login pour l&apos;application chauffeur</span>
          </div>

          <div className={s.formRow}>
            <div className={s.formGroup}>
              <label htmlFor="add_firstName">Prenom <span className={s.required}>*</span></label>
              <input type="text" id="add_firstName" name="firstName" value={formData.firstName} onChange={handleChange} placeholder="Jean" required disabled={isSubmitting} className={errors.firstName ? s.inputError : ''} />
              {errors.firstName && <span className={s.error}>{errors.firstName}</span>}
            </div>
            <div className={s.formGroup}>
              <label htmlFor="add_lastName">Nom <span className={s.required}>*</span></label>
              <input type="text" id="add_lastName" name="lastName" value={formData.lastName} onChange={handleChange} placeholder="Dupont" required disabled={isSubmitting} className={errors.lastName ? s.inputError : ''} />
              {errors.lastName && <span className={s.error}>{errors.lastName}</span>}
            </div>
          </div>

          <div className={s.formRow}>
            <div className={s.formGroup}>
              <label>Date de naissance</label>
              <InlineDatePicker value={formData.birthDate} onChange={(v) => handleChange({ target: { name: 'birthDate', value: v } })} placeholder="Naissance" />
            </div>
            <div className={s.formGroup}>
              <label htmlFor="add_nationality">Nationalite</label>
              <input type="text" id="add_nationality" name="nationality" value={formData.nationality} onChange={handleChange} placeholder="ex: Suisse" disabled={isSubmitting} />
            </div>
          </div>

          <div className={s.formGroup}>
            <label htmlFor="add_avsNumber">N° AVS</label>
            <input type="text" id="add_avsNumber" name="avsNumber" value={formData.avsNumber} onChange={handleChange} placeholder="756.XXXX.XXXX.XX" disabled={isSubmitting} />
          </div>

          <div className={s.formGroup}>
            <label htmlFor="add_email">Email <span className={s.required}>*</span></label>
            <input type="email" id="add_email" name="email" value={formData.email} onChange={handleChange} placeholder="jean.dupont@entreprise.ch" required disabled={isSubmitting} className={errors.email ? s.inputError : ''} />
            {errors.email && <span className={s.error}>{errors.email}</span>}
          </div>

          <div className={s.formGroup}>
            <label htmlFor="add_phone">Telephone</label>
            <input type="tel" id="add_phone" name="phone" value={formData.phone} onChange={handleChange} placeholder="+41 7X XXX XX XX" disabled={isSubmitting} />
          </div>

          <div className={s.formGroup}>
            <label htmlFor="add_address">Adresse</label>
            <AddressAutocomplete id="add_address" name="add_address" value={domicileAddress} onChange={handleAddressChange} onSelect={handleAddressSelect} placeholder="Adresse complete" disabled={isSubmitting} />
          </div>
        </div>

        {/* ── Acces ── */}
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <FiLock size={13} />
            <span>Acces</span>
          </div>

          <div className={s.formGroup}>
            <label htmlFor="add_password">Mot de passe <span className={s.required}>*</span></label>
            <div className={s.passwordWrap}>
              <input type={showPassword ? 'text' : 'password'} id="add_password" name="password" value={formData.password} onChange={handleChange} autoComplete="new-password" placeholder="Min. 8 caracteres" required disabled={isSubmitting} className={errors.password ? s.inputError : ''} />
              <button type="button" className={s.eyeBtn} onClick={() => setShowPassword((v) => !v)} tabIndex={-1} aria-label={showPassword ? 'Masquer' : 'Afficher'}>
                {showPassword ? <FiEyeOff size={15} /> : <FiEye size={15} />}
              </button>
            </div>
            {errors.password && <span className={s.error}>{errors.password}</span>}
            {formData.password.length > 0 && (
              <div className={s.strengthRow}>
                <div className={s.strengthBar}>
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className={s.strengthSegment} style={{ background: passwordStrength >= i ? strengthColor : '#e2e8f0' }} />
                  ))}
                </div>
                <span className={s.strengthLabel} style={{ color: strengthColor }}>{strengthLabel}</span>
              </div>
            )}
          </div>

          <div className={s.formGroup}>
            <label htmlFor="add_confirmPassword">Confirmation <span className={s.required}>*</span></label>
            <div className={s.passwordWrap}>
              <input
                type={showConfirm ? 'text' : 'password'}
                id="add_confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                autoComplete="new-password"
                placeholder="Répéter le mot de passe"
                required
                disabled={isSubmitting}
                className={
                  errors.confirmPassword ? s.inputError
                  : formData.confirmPassword && formData.confirmPassword === formData.password ? s.inputSuccess
                  : formData.confirmPassword && formData.confirmPassword !== formData.password ? s.inputError
                  : ''
                }
              />
              <button type="button" className={s.eyeBtn} onClick={() => setShowConfirm((v) => !v)} tabIndex={-1} aria-label={showConfirm ? 'Masquer' : 'Afficher'}>
                {showConfirm ? <FiEyeOff size={15} /> : <FiEye size={15} />}
              </button>
            </div>
            {errors.confirmPassword && <span className={s.error}>{errors.confirmPassword}</span>}
            {!errors.confirmPassword && formData.confirmPassword && formData.confirmPassword !== formData.password && (
              <span className={s.error}>Les mots de passe ne correspondent pas</span>
            )}
            {!errors.confirmPassword && formData.confirmPassword && formData.confirmPassword === formData.password && (
              <span className={s.successHint}>Les mots de passe correspondent</span>
            )}
          </div>
        </div>

        {/* ── Vehicule ── */}
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <FiTruck size={13} />
            <span>Vehicule</span>
          </div>
          <div className={s.formGroup}>
            <label>Véhicule assigné</label>
            {loadingVehicles ? (
              <div className={s.loadingText}>Chargement des véhicules...</div>
            ) : vehicles.length === 0 ? (
              <div className={s.warningText}>Aucun véhicule disponible. Créez d&apos;abord un véhicule dans la gestion de flotte.</div>
            ) : (
              <PortalChipDropdown
                options={[
                  { value: '', label: 'Aucun véhicule' },
                  ...vehicles.map((v) => ({ value: String(v.id), label: `${v.model} - ${v.license_plate}${v.brand ? ` (${v.brand})` : ''}` })),
                ]}
                value={formData.vehicleId}
                onChange={(v) => handleChange({ target: { name: 'vehicleId', value: v } })}
                disabled={isSubmitting}
                placeholder="Aucun véhicule"
              />
            )}
            <span className={s.hint}>Peut être assigné ultérieurement</span>
          </div>
        </div>

        {/* ── Permis et contrat ── */}
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <FiShield size={13} />
            <span>Permis et contrat</span>
          </div>

          <div className={s.formRow}>
            <div className={s.formGroup}>
              <label>Contrat</label>
              <PortalChipDropdown
                options={CONTRACT_OPTIONS}
                value={formData.contractType}
                onChange={(v) => handleChange({ target: { name: 'contractType', value: v } })}
                disabled={isSubmitting}
                placeholder="CDI"
              />
            </div>
            <div className={s.formGroup}>
              <label>Debut emploi</label>
              <InlineDatePicker value={formData.employmentStartDate} onChange={(v) => handleChange({ target: { name: 'employmentStartDate', value: v } })} placeholder="Début" />
            </div>
          </div>

          <div className={s.formRow}>
            <div className={s.formGroup}>
              <label>Validite permis</label>
              <InlineDatePicker value={formData.licenseValidUntil} onChange={(v) => handleChange({ target: { name: 'licenseValidUntil', value: v } })} placeholder="Permis" />
            </div>
            <div className={s.formGroup}>
              <label>Validite medical</label>
              <InlineDatePicker value={formData.medicalValidUntil} onChange={(v) => handleChange({ target: { name: 'medicalValidUntil', value: v } })} placeholder="Médical" />
            </div>
          </div>
        </div>

        {/* ── Contact d'urgence ── */}
        <div className={s.section}>
          <div className={s.sectionHeader}>
            <FiAlertTriangle size={13} />
            <span>Contact d&apos;urgence</span>
          </div>
          <div className={s.formRow}>
            <div className={s.formGroup}>
              <label htmlFor="add_emergencyContactName">Nom</label>
              <input type="text" id="add_emergencyContactName" name="emergencyContactName" value={formData.emergencyContactName} onChange={handleChange} placeholder="Nom du contact" disabled={isSubmitting} />
            </div>
            <div className={s.formGroup}>
              <label htmlFor="add_emergencyContactPhone">Telephone</label>
              <input type="tel" id="add_emergencyContactPhone" name="emergencyContactPhone" value={formData.emergencyContactPhone} onChange={handleChange} placeholder="+41 XX XXX XX XX" disabled={isSubmitting} />
            </div>
          </div>
        </div>

        {/* ── Statut ── */}
        <div className={s.section}>
          <label className={`${ntStyles.notifRow} ${s.toggleRow}`} htmlFor="add_isActive">
            <div className={ntStyles.notifInfo}>
              <span className={ntStyles.notifLabel}>Chauffeur actif</span>
              <span className={ntStyles.notifHint}>Les chauffeurs inactifs n'apparaissent pas dans les sélections</span>
            </div>
            <div className={ntStyles.miniToggle}>
              <input id="add_isActive" type="checkbox" name="isActive" checked={formData.isActive} onChange={handleChange} disabled={isSubmitting} />
              <span className={ntStyles.miniSlider} />
            </div>
          </label>
        </div>

        {/* Footer */}
        <div className={s.footer}>
          <button type="button" onClick={onClose} className={s.cancelButton} disabled={isSubmitting}>
            Annuler
          </button>
          <button type="submit" className={s.submitButton} disabled={isSubmitting}>
            <FiUserPlus size={14} />
            {isSubmitting ? 'Ajout en cours...' : 'Ajouter le chauffeur'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default AddDriverForm;
