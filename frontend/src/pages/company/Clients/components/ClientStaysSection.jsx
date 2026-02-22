// frontend/src/pages/company/Clients/components/ClientStaysSection.jsx
import React, { useState, useEffect, useCallback } from 'react';
import {
  FiActivity,
  FiPlus,
  FiCheck,
  FiEdit2,
  FiChevronDown,
} from 'react-icons/fi';
import {
  fetchClientStays,
  createClientStay,
  updateClientStay,
  closeClientStay,
} from '../../../../services/companyService';
import { fetchClinicBillingMappings } from '../../../../services/settingsService';
import styles from './ClientStaysSection.module.css';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';

function ClinicChipDropdown({ clinics, value, onChange }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);

  React.useEffect(() => {
    if (!open) return;
    const onClick = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const selected = clinics.find((c) => String(c.id) === String(value));

  return (
    <div className={styles.chipDrop} ref={ref}>
      <button
        type="button"
        className={`${styles.chipBtn} ${selected ? styles.chipBtnActive : ''}`}
        onClick={() => setOpen((p) => !p)}
      >
        <span className={styles.chipText}>{selected?.name || 'Sélectionner une clinique'}</span>
        <FiChevronDown size={11} className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`} />
      </button>
      {open && (
        <div className={styles.chipMenu}>
          <button
            type="button"
            className={`${styles.chipOption} ${!value ? styles.chipOptionActive : ''}`}
            onClick={() => { onChange(''); setOpen(false); }}
          >
            Aucune
          </button>
          {clinics.map((c) => (
            <button
              key={c.id}
              type="button"
              className={`${styles.chipOption} ${String(c.id) === String(value) ? styles.chipOptionActive : ''}`}
              onClick={() => { onChange(String(c.id)); setOpen(false); }}
            >
              {c.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const ClientStaysSection = ({ clientId }) => {
  const [stays, setStays] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [editingStay, setEditingStay] = useState(null);
  const [clinics, setClinics] = useState([]);
  const [formData, setFormData] = useState({
    company_id: '',
    start_date: '',
    end_date: '',
    notes: '',
  });

  const loadStays = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetchClientStays(clientId);
      setStays(response.data || []);
    } catch (err) {
      setError(err.response?.data?.error || 'Erreur lors du chargement des séjours');
    } finally {
      setLoading(false);
    }
  }, [clientId]);

  const loadClinics = useCallback(async () => {
    try {
      const response = await fetchClinicBillingMappings();
      const mappings = response.data || [];
      const uniqueClinics = [];
      const seen = new Set();
      mappings.forEach((m) => {
        if (m.clinic_company_id && !seen.has(m.clinic_company_id)) {
          seen.add(m.clinic_company_id);
          uniqueClinics.push({
            id: m.clinic_company_id,
            name: m.clinic_company_name,
          });
        }
      });
      setClinics(uniqueClinics);
    } catch (err) {
      console.error('Erreur lors du chargement des cliniques:', err);
    }
  }, []);

  useEffect(() => {
    loadStays();
    loadClinics();
  }, [loadStays, loadClinics]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setError(null);
      const payload = {
        company_id: parseInt(formData.company_id),
        start_date: formData.start_date,
        end_date: formData.end_date || null,
        notes: formData.notes || null,
      };

      if (editingStay) {
        await updateClientStay(editingStay.id, payload);
      } else {
        const activeStay = getActiveStay();
        if (activeStay) {
          await closeClientStay(activeStay.id, formData.start_date);
        }
        await createClientStay(clientId, payload);
      }

      await loadStays();
      setShowForm(false);
      setEditingStay(null);
      setFormData({
        company_id: '',
        start_date: '',
        end_date: '',
        notes: '',
      });
    } catch (err) {
      setError(err.response?.data?.error || 'Erreur lors de la sauvegarde');
    }
  };

  const handleEdit = (stay) => {
    setEditingStay(stay);
    setFormData({
      company_id: String(stay.company_id),
      start_date: stay.start_date ? stay.start_date.split('T')[0] : '',
      end_date: stay.end_date ? stay.end_date.split('T')[0] : '',
      notes: stay.notes || '',
    });
    setShowForm(true);
  };

  const handleClose = async (stayId, endDate = null) => {
    const message = endDate 
      ? `Voulez-vous fermer ce séjour avec la date de fin ${new Date(endDate).toLocaleDateString('fr-FR')} ?`
      : 'Voulez-vous fermer ce séjour (définir la date de fin à aujourd\'hui) ?';
    
    if (window.confirm(message)) {
      try {
        await closeClientStay(stayId, endDate);
        await loadStays();
      } catch (err) {
        setError(err.response?.data?.error || 'Erreur lors de la fermeture');
      }
    }
  };

  const getActiveStay = () => {
    return stays.find((s) => s.status === 'active' && !s.end_date);
  };

  const activeStay = getActiveStay();

  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>
          <FiActivity size={14} className={styles.sectionIcon} />
          Hospitalisation
        </h3>
        {!showForm && (
          <button
            type="button"
            onClick={() => {
              setShowForm(true);
              setEditingStay(null);
              setFormData({
                company_id: '',
                start_date: new Date().toISOString().split('T')[0],
                end_date: '',
                notes: '',
              });
            }}
            className={styles.addButton}
            title="Ajouter un nouveau séjour d'hospitalisation"
          >
            <FiPlus size={14} />
            Nouveau sejour
          </button>
        )}
      </div>

      {activeStay && (
        <div className={styles.activeStayBadge}>
          <FiActivity size={20} className={styles.badgeIcon} />
          <div className={styles.badgeContent}>
            <strong>Client actuellement hospitalise</strong>
            <div className={styles.badgeDetails}>
              {activeStay.company_name} \u2014 Depuis le{' '}
              {new Date(activeStay.start_date).toLocaleDateString('fr-FR')}
            </div>
          </div>
        </div>
      )}

      {showForm && (
        <div className={styles.form}>
          <div className={styles.formGroup}>
            <label htmlFor="company_id" className={styles.label}>
              Clinique / Etablissement *
            </label>
            <ClinicChipDropdown
              clinics={clinics}
              value={formData.company_id}
              onChange={(v) => setFormData({ ...formData, company_id: v })}
            />
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label className={styles.label}>Date de debut *</label>
              <InlineDatePicker
                value={formData.start_date}
                onChange={(v) => setFormData({ ...formData, start_date: v })}
                placeholder="Début"
              />
            </div>

            <div className={styles.formGroup}>
              <label className={styles.label}>Date de fin (optionnel)</label>
              <InlineDatePicker
                value={formData.end_date}
                onChange={(v) => setFormData({ ...formData, end_date: v })}
                placeholder="Fin"
              />
              <small className={styles.hint}>
                Laisser vide si le sejour est en cours
              </small>
            </div>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="notes" className={styles.label}>
              Notes (optionnel)
            </label>
            <textarea
              id="notes"
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              className={styles.textarea}
              rows={3}
              placeholder="Informations complementaires sur le sejour..."
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <div className={styles.formActions}>
            <button
              type="button"
              onClick={() => {
                setShowForm(false);
                setEditingStay(null);
                setFormData({
                  company_id: '',
                  start_date: '',
                  end_date: '',
                  notes: '',
                });
              }}
              className={styles.cancelButton}
            >
              Annuler
            </button>
            <button 
              type="button" 
              onClick={handleSubmit}
              className={styles.saveButton}
            >
              {editingStay ? 'Modifier' : 'Creer'} le sejour
            </button>
          </div>
        </div>
      )}

      {!showForm && (
        <>
          {loading ? (
            <div className={styles.loading}>Chargement des sejours...</div>
          ) : stays.length === 0 ? (
            <div className={styles.emptyState}>
              Aucun sejour enregistre. Cliquez sur &laquo; Nouveau sejour &raquo; pour en ajouter un.
            </div>
          ) : (
            <div className={styles.staysList}>
              {stays.map((stay) => (
                <div key={stay.id} className={styles.stayCard}>
                  <div className={styles.stayHeader}>
                    <div className={styles.stayInfo}>
                      <strong>{stay.company_name || `Clinique #${stay.company_id}`}</strong>
                      <div className={styles.stayDates}>
                        Du {new Date(stay.start_date).toLocaleDateString('fr-FR')}
                        {stay.end_date
                          ? ` au ${new Date(stay.end_date).toLocaleDateString('fr-FR')}`
                          : ' (en cours)'}
                      </div>
                    </div>
                    <div className={styles.stayActions}>
                      {!stay.end_date && (
                        <button
                          type="button"
                          onClick={() => handleClose(stay.id)}
                          className={styles.closeStayButton}
                          title="Fermer le sejour"
                        >
                          <FiCheck size={12} />
                          Fermer
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => handleEdit(stay)}
                        className={styles.editButton}
                        title="Modifier"
                      >
                        <FiEdit2 size={12} />
                      </button>
                    </div>
                  </div>
                  {stay.notes && (
                    <div className={styles.stayNotes}>
                      <em>{stay.notes}</em>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ClientStaysSection;
