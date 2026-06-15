// pages/institution/Settings/components/InstitutionProfileTab.jsx
/**
 * Onglet Profil : affiche et permet d'éditer les infos de l'institution.
 * - Visible par tous les rôles (lecture)
 * - Éditable uniquement par institution_admin
 *
 * Le Profil = identité de l'institution (pas facturation, pas notifications).
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { FaSave, FaBuilding, FaUpload, FaTrash } from 'react-icons/fa';
import { useInstitutionMe, useUpdateInstitution, institutionQueryKeys } from '../../../../hooks/useInstitutionData';
import { useQueryClient } from '@tanstack/react-query';
import { uploadInstitutionLogo, deleteInstitutionLogo } from '../../../../services/institutionService';
import { isAdmin } from '../../../../utils/institutionPermissions';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import resolveLogoUrl from '../../../../utils/resolveLogoUrl';
import { toast } from 'sonner';
import ChipSelect from './ChipSelect';
import styles from '../InstitutionSettings.module.css';

const FIELDS = ['name', 'institution_type', 'address', 'contact_email', 'contact_phone', 'notes'];

const INSTITUTION_TYPES = [
  { value: 'clinic', label: 'Clinique' },
  { value: 'ems', label: 'EMS' },
  { value: 'imad', label: 'IMAD' },
  { value: 'hospital', label: 'Hôpital' },
  { value: 'curatelle', label: 'Curatelle' },
  { value: 'other', label: 'Autre' },
];

const InstitutionProfileTab = () => {
  const { data: meData, isLoading } = useInstitutionMe();
  const updateMutation = useUpdateInstitution();
  const queryClient = useQueryClient();

  const institutionRole = meData?.institution_role;
  const canEdit = isAdmin(institutionRole);

  const fileInputRef = useRef(null);
  const [logoPreview, setLogoPreview] = useState(null);
  const [logoBusy, setLogoBusy] = useState(false);

  const [form, setForm] = useState({
    name: '',
    institution_type: '',
    address: '',
    contact_email: '',
    contact_phone: '',
    notes: '',
  });

  // Ref pour tracker la version des données serveur
  const lastSyncedRef = useRef(null);

  // Synchroniser le formulaire avec les données serveur
  // chaque fois que meData change (initial load OU après invalidation post-save)
  useEffect(() => {
    if (!meData) return;

    // Calculer une "empreinte" des données pour détecter un vrai changement
    const serverFingerprint = FIELDS.map((f) => meData[f] || '').join('|');
    if (serverFingerprint === lastSyncedRef.current) return;

    setForm({
      name: meData.name || '',
      institution_type: meData.institution_type || '',
      address: meData.address || '',
      contact_email: meData.contact_email || '',
      contact_phone: meData.contact_phone || '',
      notes: meData.notes || '',
    });
    lastSyncedRef.current = serverFingerprint;
  }, [meData]);

  useEffect(() => {
    if (!meData) return;
    const resolved = resolveLogoUrl(meData.logo_url);
    setLogoPreview(resolved || null);
  }, [meData?.logo_url]);

  // Dirty state : détecter si des champs ont changé
  const isDirty = useMemo(() => {
    if (!meData) return false;
    return FIELDS.some((f) => (form[f]?.trim() || '') !== (meData[f] || ''));
  }, [form, meData]);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    const payload = {};
    for (const field of FIELDS) {
      const current = form[field]?.trim() || '';
      const original = meData?.[field] || '';
      if (current !== original) {
        payload[field] = current || null;
      }
    }
    // name ne peut pas être null
    if (payload.name === null) {
      toast.error('Le nom de l\'institution est obligatoire');
      return;
    }

    if (Object.keys(payload).length === 0) {
      toast.info('Aucune modification détectée');
      return;
    }

    console.log('[InstitutionProfileTab] Saving:', payload);

    try {
      const result = await updateMutation.mutateAsync(payload);
      console.log('[InstitutionProfileTab] Save success:', result);

      // Forcer la re-synchronisation avec les données serveur
      lastSyncedRef.current = null;

      toast.success('Les informations de l\'institution ont été mises à jour.');
    } catch (err) {
      console.error('[InstitutionProfileTab] Save error:', err);
      const msg =
        err?.response?.data?.error ||
        err?.response?.data?.details ||
        err?.message ||
        'Erreur lors de la mise à jour';
      toast.error(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
  };

  const onPickLogoFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'image/svg+xml'];
    if (!allowed.includes(file.type)) {
      toast.error('Format non supporté (PNG, JPG ou SVG).');
      return;
    }
    if (file.size > 2 * 1024 * 1024) {
      toast.error('Fichier trop volumineux (max 2 Mo).');
      return;
    }

    const localUrl = URL.createObjectURL(file);
    setLogoPreview(localUrl);
    setLogoBusy(true);

    try {
      const result = await uploadInstitutionLogo(file);
      if (result?.logo_url) {
        setLogoPreview(resolveLogoUrl(result.logo_url) || null);
      }
      lastSyncedRef.current = null;
      await queryClient.invalidateQueries({ queryKey: institutionQueryKeys.me() });
      toast.success('Logo mis à jour.');
    } catch (err) {
      const msg =
        err?.response?.data?.error ||
        err?.message ||
        "Échec de l'upload du logo.";
      toast.error(typeof msg === 'string' ? msg : JSON.stringify(msg));
      setLogoPreview(resolveLogoUrl(meData?.logo_url) || null);
    } finally {
      setLogoBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const onRemoveLogo = async () => {
    if (!meData?.logo_url) return;
    setLogoBusy(true);
    try {
      await deleteInstitutionLogo();
      setLogoPreview(null);
      lastSyncedRef.current = null;
      await queryClient.invalidateQueries({ queryKey: institutionQueryKeys.me() });
      toast.success('Logo supprimé.');
    } catch (err) {
      const msg =
        err?.response?.data?.error ||
        err?.message ||
        'Impossible de supprimer le logo.';
      toast.error(typeof msg === 'string' ? msg : JSON.stringify(msg));
      setLogoPreview(resolveLogoUrl(meData?.logo_url) || null);
    } finally {
      setLogoBusy(false);
    }
  };

  if (isLoading) {
    return <p>Chargement...</p>;
  }

  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3>
          <FaBuilding aria-hidden="true" />
          Informations de l'institution
        </h3>
        <p>
          Ces informations identifient votre institution dans le portail et sont
          utilisées pour les échanges, notifications et documents associés aux
          transports.
        </p>
      </div>

      <div className={styles.profileForm}>
        <div className={styles.profileIdentityRow}>
          <div className={styles.profileIdentityFields}>
            {/* Nom */}
            <div className={styles.field}>
              <label>Nom de l'institution *</label>
              <input
                type="text"
                value={form.name}
                onChange={(e) => handleChange('name', e.target.value)}
                disabled={!canEdit}
                placeholder="Ex : Clinique les Hauts d'Anières"
              />
              <span className={styles.fieldHint}>
                Nom officiel tel qu'il apparaît dans le portail et les
                communications.
              </span>
            </div>

            {/* Type d'institution */}
            <div className={styles.field}>
              <label htmlFor="institution-type">Type d'institution</label>
              {canEdit ? (
                <ChipSelect
                  id="institution-type"
                  name="institution_type"
                  ariaLabel="Type d'institution"
                  block
                  placeholder="Sélectionner un type"
                  value={form.institution_type || ''}
                  options={INSTITUTION_TYPES}
                  onChange={(val) => handleChange('institution_type', val)}
                />
              ) : (
                <input
                  type="text"
                  value={
                    INSTITUTION_TYPES.find(
                      (t) => t.value === meData?.institution_type?.toLowerCase()
                    )?.label ||
                    meData?.institution_type ||
                    ''
                  }
                  disabled
                  className={styles.readonlyField}
                />
              )}
              <span className={styles.fieldHint}>
                Détermine le fonctionnement du portail (équipes, accès patients, facturation).
              </span>
            </div>
          </div>

          {/* Logo institution — colonne droite */}
          <div className={`${styles.field} ${styles.profileIdentityLogo}`}>
            <label>Logo de l'institution</label>
            <div className={styles.logoColumn}>
              <div className={styles.logoBox}>
                {logoPreview ? (
                  <img
                    src={logoPreview}
                    alt="Logo de l'institution"
                    className={styles.logoPreview}
                    loading="lazy"
                  />
                ) : (
                  <div className={styles.logoPlaceholder}>
                    <span>Aucun logo</span>
                  </div>
                )}
              </div>
              {canEdit ? (
                <div className={styles.logoActions}>
                  <span className={styles.fieldHint}>
                    PNG, JPG ou SVG — 2 Mo max. Affiché sur les bons de transport.
                  </span>
                  <div className={styles.logoActionsRow}>
                    <button
                      type="button"
                      className={styles.logoActionBtn}
                      onClick={() => fileInputRef.current?.click()}
                      disabled={logoBusy}
                    >
                      <FaUpload aria-hidden="true" />{' '}
                      {logoBusy
                        ? 'Téléversement...'
                        : logoPreview
                          ? 'Remplacer'
                          : 'Ajouter'}
                    </button>
                    {meData?.logo_url && (
                      <button
                        type="button"
                        className={`${styles.logoActionBtn} ${styles.logoActionDanger}`}
                        onClick={onRemoveLogo}
                        disabled={logoBusy}
                      >
                        <FaTrash aria-hidden="true" /> Supprimer
                      </button>
                    )}
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/jpg,image/svg+xml"
                    className={styles.logoFileInput}
                    onChange={onPickLogoFile}
                  />
                </div>
              ) : (
                <span className={styles.fieldHint}>
                  Affiché sur les bons de transport.
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Email + Téléphone */}
        <div className={styles.fieldRow}>
          <div className={styles.field}>
            <label>Email de contact</label>
            <input
              type="email"
              value={form.contact_email}
              onChange={(e) => handleChange('contact_email', e.target.value)}
              disabled={!canEdit}
              placeholder="contact@clinique.ch"
            />
            <span className={styles.fieldHint}>
              Adresse pour les communications générales. Ce n'est pas l'email de
              facturation ni de notification.
            </span>
          </div>
          <div className={styles.field}>
            <label>Téléphone</label>
            <input
              type="text"
              value={form.contact_phone}
              onChange={(e) => handleChange('contact_phone', e.target.value)}
              disabled={!canEdit}
              placeholder="+41 22 123 45 67"
            />
            <span className={styles.fieldHint}>
              Numéro de contact principal de l'institution.
            </span>
          </div>
        </div>

        {/* Adresse */}
        <div className={styles.field}>
          <label>Adresse</label>
          {canEdit ? (
            <AddressAutocomplete
              name="institution_address"
              inputId="institution_address"
              value={form.address || ''}
              onChange={(e) => handleChange('address', e.target.value)}
              onSelect={(item) => {
                const parts = [
                  item.address || item.label || '',
                  item.postcode || '',
                  item.city || '',
                ].filter(Boolean);
                handleChange('address', parts.join(', '));
              }}
              placeholder="Tapez pour rechercher une adresse..."
            />
          ) : (
            <input
              type="text"
              value={form.address}
              disabled
              placeholder="Rue de l'Exemple 1, 1200 Genève"
            />
          )}
          <span className={styles.fieldHint}>
            Adresse officielle de l'institution.
          </span>
        </div>

        {/* Notes internes */}
        <div className={styles.field}>
          <label>Notes internes</label>
          <textarea
            value={form.notes}
            onChange={(e) => handleChange('notes', e.target.value)}
            disabled={!canEdit}
            rows={3}
            placeholder="Notes visibles uniquement par votre équipe..."
          />
          <span className={styles.fieldHint}>
            Visibles uniquement par les utilisateurs de votre institution. Ces
            informations ne sont jamais transmises aux transporteurs.
          </span>
        </div>

        {/* Bouton Enregistrer */}
        {canEdit && (
          <button
            className={styles.saveBtn}
            onClick={handleSave}
            disabled={updateMutation.isPending || !isDirty}
          >
            <FaSave />{' '}
            {updateMutation.isPending ? 'Enregistrement...' : 'Enregistrer'}
          </button>
        )}
      </div>
    </div>
  );
};

export default InstitutionProfileTab;
