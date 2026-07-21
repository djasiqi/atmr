// pages/institution/Patients/InstitutionPatients.jsx
/**
 * Gestion des patients — card-row design professionnel
 *
 * Sections formulaire:
 * 1. Identité (civilité, nom, prénom, date de naissance, téléphone)
 * 2. Adresse (rue, NPA, ville)
 * 3. Accès & logistique (code porte, étage, notes d'accès, résidence)
 * 4. Informations administratives (AVS, assurance, curatelle)
 * 5. Notes internes
 */

import React, { useState, useMemo, useCallback } from 'react';
import { FaPlus, FaSearch, FaPhone, FaMapMarkerAlt, FaBirthdayCake, FaHome, FaGavel } from 'react-icons/fa';
import PatientDetailPanel from './PatientDetailPanel';
import PatientFormModal from './PatientFormModal';
import { useInstitutionPatients, useInstitutionMe } from '../../../hooks/useInstitutionData';
import { canManageRequests } from '../../../utils/institutionPermissions';
import s from './InstitutionPatients.module.css';

// ─── Constants ─────────────────────────────────────────────
const GENDER_SHORT = { HOMME: 'M.', FEMME: 'Mme', AUTRE: '' };

const GUARDIANSHIP_TYPE_LABELS = {
  opad: 'OPAD / SPAd',
  curatorship: 'Curateur professionnel',
  lawyer: 'Avocat',
  family: 'Famille',
  other: 'Autre',
};

// ─── Helpers ───────────────────────────────────────────────
const fmtDate = (d) => {
  if (!d) return '—';
  return new Date(d).toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit', year: 'numeric' });
};

const getInitials = (first, last) => {
  const f = (first || '').charAt(0).toUpperCase();
  const l = (last || '').charAt(0).toUpperCase();
  return f + l || '?';
};

// ─── Component ─────────────────────────────────────────────
const InstitutionPatients = () => {
  const { data: meData } = useInstitutionMe();
  const { data: patientsData, isLoading, error } = useInstitutionPatients({
    fetchAll: true,
    per_page: 500,
  });

  const institutionRole = meData?.institution_role;
  const canManage = canManageRequests(institutionRole);

  const [searchQuery, setSearchQuery] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingPatient, setEditingPatient] = useState(null);
  const [selectedPatientId, setSelectedPatientId] = useState(null);

  const patients = useMemo(
    () => patientsData?.patients || patientsData?.items || [],
    [patientsData]
  );

  const totalPatients = patientsData?.total ?? patients.length;

  const filteredPatients = useMemo(() => {
    if (!searchQuery) return patients;
    const q = searchQuery.toLowerCase();
    return patients.filter(p =>
      p.first_name?.toLowerCase().includes(q) ||
      p.last_name?.toLowerCase().includes(q) ||
      p.phone?.includes(q) ||
      p.city?.toLowerCase().includes(q) ||
      p.residence_name?.toLowerCase().includes(q) ||
      p.address?.toLowerCase().includes(q)
    );
  }, [patients, searchQuery]);

  const displayCount = searchQuery ? filteredPatients.length : totalPatients;

  // Group alphabetically by last name
  const grouped = useMemo(() => {
    const sorted = [...filteredPatients].sort((a, b) =>
      (a.last_name || '').localeCompare(b.last_name || '', 'fr')
    );
    const groups = {};
    for (const p of sorted) {
      const letter = (p.last_name || '?').charAt(0).toUpperCase();
      if (!groups[letter]) groups[letter] = [];
      groups[letter].push(p);
    }
    return groups;
  }, [filteredPatients]);

  const openCreateModal = () => {
    setEditingPatient(null);
    setShowCreateModal(true);
  };

  const _openEditModal = (patient) => {
    setEditingPatient(patient);
    setShowCreateModal(true);
  };

  // ── Panel latéral ──
  const selectedPatient = useMemo(
    () => (selectedPatientId ? patients.find(p => p.id === selectedPatientId) : null),
    [patients, selectedPatientId]
  );
  const panelOpen = !!selectedPatient;

  const handleSelectPatient = useCallback((id) => {
    setSelectedPatientId(prev => (prev === id ? null : id));
  }, []);

  const handleClosePanel = useCallback(() => setSelectedPatientId(null), []);

  return (
    <div className={`${s.masterDetail} ${panelOpen ? s.masterDetailOpen : ''}`}>
      {/* ════ Colonne liste ════ */}
      <div className={s.listColumn}>

      {/* ═══ TOOLBAR ═══ */}
      <div className={s.toolbar}>
        <div className={s.searchBox}>
          <FaSearch className={s.searchIcon} />
          <input
            type="text"
            placeholder="Rechercher par nom, téléphone, ville ou résidence..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className={s.toolbarRight}>
          {displayCount > 0 && (
            <span className={s.patientCount}>{displayCount} patient{displayCount > 1 ? 's' : ''}</span>
          )}
          {canManage && (
            <button className={s.addBtn} onClick={openCreateModal}>
              <FaPlus size={11} /> Nouveau patient
            </button>
          )}
        </div>
      </div>

      {/* ═══ LIST ═══ */}
      {isLoading ? (
        <div className={s.loading}>Chargement...</div>
      ) : error ? (
        <div className={s.error}>Erreur : {error.message}</div>
      ) : filteredPatients.length === 0 ? (
        <div className={s.empty}>
          <p>Aucun patient trouvé</p>
          <p className={s.emptyHint}>
            {searchQuery ? 'Essayez avec d\'autres termes.' : 'Commencez par créer un patient.'}
          </p>
          {canManage && !searchQuery && (
            <button className={s.emptyBtn} onClick={openCreateModal}>
              <FaPlus size={11} /> Créer un patient
            </button>
          )}
        </div>
      ) : (
        Object.entries(grouped).map(([letter, items]) => (
          <div key={letter} className={s.letterGroup}>
            <div className={s.letterLabel}>{letter}</div>

            {items.map((patient) => {
              const fullAddr = [patient.address, patient.postal_code, patient.city].filter(Boolean).join(', ');

              return (
                <div
                  key={patient.id}
                  className={`${s.patientCard} ${selectedPatientId === patient.id ? s.patientCardSelected : ''}`}
                  onClick={() => handleSelectPatient(patient.id)}
                  role="button"
                  tabIndex={0}
                >
                  {/* Avatar */}
                  <div className={s.avatar}>
                    {getInitials(patient.first_name, patient.last_name)}
                  </div>

                  <div className={s.cardBody}>
                    {/* Name */}
                    <div className={s.cardName}>
                      <div className={s.nameMain}>
                        {patient.last_name} {patient.first_name}
                      </div>
                      {patient.gender && (
                        <div className={s.nameGender}>{GENDER_SHORT[patient.gender] || ''}</div>
                      )}
                    </div>

                    {/* Info pills */}
                    <div className={s.cardInfo}>
                      {patient.dob && (
                        <span className={s.infoPill}>
                          <FaBirthdayCake className={s.infoPillIcon} />
                          {fmtDate(patient.dob)}
                        </span>
                      )}
                      {patient.phone && (
                        <span className={s.infoPill}>
                          <FaPhone className={s.infoPillIcon} />
                          {patient.phone}
                        </span>
                      )}
                      {fullAddr && (
                        <span className={s.infoPill} title={fullAddr}>
                          <FaMapMarkerAlt className={s.infoPillIcon} />
                          {fullAddr.length > 40 ? fullAddr.substring(0, 40) + '...' : fullAddr}
                        </span>
                      )}
                      {patient.residence_name && (
                        <span className={s.infoPill}>
                          <FaHome className={s.infoPillIcon} />
                          {patient.residence_name}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Badge curatelle à droite */}
                  {patient.has_guardianship && (
                    <span
                      className={`${s.infoPill} ${s.infoPillCuratelle} ${s.cardBadgeRight}`}
                      title={patient.guardianship_type ? GUARDIANSHIP_TYPE_LABELS[patient.guardianship_type] : 'Sous curatelle'}
                    >
                      <FaGavel className={s.infoPillIcon} />
                      {patient.guardianship_type
                        ? GUARDIANSHIP_TYPE_LABELS[patient.guardianship_type]
                        : 'Curatelle'}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ))
      )}

      </div>{/* fin listColumn */}

      {/* ════ Colonne détail (panel latéral) ════ */}
      {panelOpen && (
        <div className={s.detailColumn}>
          <PatientDetailPanel
            patient={selectedPatient}
            onClose={handleClosePanel}
          />
        </div>
      )}

      {/* ═══ MODAL ═══ */}
      {showCreateModal && (
        <PatientFormModal
          onClose={() => setShowCreateModal(false)}
          onSaved={(patient) => {
            setShowCreateModal(false);
            if (patient?.id) setSelectedPatientId(patient.id);
          }}
          editingPatient={editingPatient}
        />
      )}
    </div>
  );
};

export default InstitutionPatients;
