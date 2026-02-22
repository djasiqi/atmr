// frontend/src/pages/company/Settings/tabs/VehiclesTab.jsx
import React, { useState, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import { FiTruck, FiPlus, FiCheck, FiX, FiTrash2 } from 'react-icons/fi';
import {
  fetchCompanyVehicles,
  createVehicle,
  updateVehicle,
  deleteVehicle,
} from '../../../../services/companyService';
import styles from '../CompanySettings.module.css';
import vehicleStyles from './VehiclesTab.module.css';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';

const EMPTY_ROW = () => ({
  _tempId: `new_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
  model: '',
  license_plate: '',
  year: '',
  seats: '',
  wheelchair_accessible: false,
  is_active: true,
  insurance_company_name: '',
  inspection_expires_at: '',
  tachograph_expires_at: '',
});

function toDateOnly(dateStr) {
  if (!dateStr) return null;
  return dateStr.includes('T') ? dateStr.split('T')[0] : dateStr;
}

function formatDateCH(isoDate) {
  if (!isoDate) return '\u2014';
  const [y, m, d] = isoDate.split('-');
  if (!y || !m || !d) return isoDate;
  return `${d}.${m}.${y}`;
}

/**
 * Badge pour une date de controle avec duree de validite.
 * @param {string|null} dateStr  Date du controle (YYYY-MM-DD ou ISO)
 * @param {number}      validityYears  Duree de validite en annees
 */
function getControlBadge(dateStr, validityYears) {
  const iso = toDateOnly(dateStr);
  if (!iso) return { label: '\u2014', cls: vehicleStyles.dateBadgeNone };

  const controlDate = new Date(iso + 'T00:00:00');
  const expiryDate = new Date(controlDate);
  expiryDate.setFullYear(expiryDate.getFullYear() + validityYears);

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const warnDate = new Date(expiryDate);
  warnDate.setDate(warnDate.getDate() - 30);

  const label = formatDateCH(iso);
  if (today >= expiryDate) return { label, cls: vehicleStyles.dateBadgeExpired };
  if (today >= warnDate) return { label, cls: vehicleStyles.dateBadgeWarning };
  return { label, cls: vehicleStyles.dateBadgeValid };
}

const VehiclesTab = forwardRef(({ isEditing }, ref) => {
  const [vehicles, setVehicles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [message, setMessage] = useState(''); // eslint-disable-line no-unused-vars

  const [rowEdits, setRowEdits] = useState({});
  const [newRows, setNewRows] = useState([]);
  const [deletedIds, setDeletedIds] = useState([]);

  const loadVehicles = async () => {
    try {
      setLoading(true);
      setError('');
      const data = await fetchCompanyVehicles();
      setVehicles(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Erreur chargement véhicules:', err);
      setError('Impossible de charger les véhicules');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadVehicles();
  }, []);

  useEffect(() => {
    if (isEditing) {
      const edits = {};
      vehicles.forEach((v) => {
        edits[v.id] = {
          model: v.model || '',
          license_plate: v.license_plate || '',
          year: v.year || '',
          seats: v.seats || '',
          wheelchair_accessible: v.wheelchair_accessible || false,
          is_active: v.is_active !== undefined ? v.is_active : true,
          inspection_expires_at: toDateOnly(v.inspection_expires_at) || '',
          tachograph_expires_at: toDateOnly(v.tachograph_expires_at) || '',
          insurance_company_name: v.insurance_company_name || '',
        };
      });
      setRowEdits(edits);
      setNewRows([]);
      setDeletedIds([]);
    } else {
      setRowEdits({});
      setNewRows([]);
      setDeletedIds([]);
    }
  }, [isEditing, vehicles]);

  const updateField = useCallback((id, field, value) => {
    setRowEdits((prev) => ({
      ...prev,
      [id]: { ...prev[id], [field]: value },
    }));
  }, []);

  const updateNewRow = useCallback((tempId, field, value) => {
    setNewRows((prev) =>
      prev.map((r) => (r._tempId === tempId ? { ...r, [field]: value } : r))
    );
  }, []);

  const handleAddRow = useCallback(() => {
    setNewRows((prev) => [...prev, EMPTY_ROW()]);
  }, []);

  const handleRemoveNewRow = useCallback((tempId) => {
    setNewRows((prev) => prev.filter((r) => r._tempId !== tempId));
  }, []);

  const handleMarkDelete = useCallback((vehicleId) => {
    setDeletedIds((prev) => [...prev, vehicleId]);
  }, []);

  const _handleUnmarkDelete = useCallback((vehicleId) => {
    setDeletedIds((prev) => prev.filter((id) => id !== vehicleId));
  }, []);

  useImperativeHandle(ref, () => ({
    async save() {
      const errors = [];

      for (const id of deletedIds) {
        try {
          await deleteVehicle(id, false);
        } catch (err) {
          errors.push(err);
        }
      }

      for (const [id, edits] of Object.entries(rowEdits)) {
        if (deletedIds.includes(id)) continue;
        const payload = {
          model: edits.model.trim(),
          license_plate: edits.license_plate.trim().toUpperCase(),
          wheelchair_accessible: edits.wheelchair_accessible,
          is_active: edits.is_active,
          insurance_company_name: edits.insurance_company_name?.trim() || null,
        };
        if (edits.year && String(edits.year).trim()) payload.year = parseInt(edits.year);
        if (edits.seats && String(edits.seats).trim()) payload.seats = parseInt(edits.seats);
        payload.inspection_expires_at = edits.inspection_expires_at || null;
        payload.tachograph_expires_at = edits.tachograph_expires_at || null;
        try {
          await updateVehicle(id, payload);
        } catch (err) {
          errors.push(err);
        }
      }

      for (const row of newRows) {
        if (!row.model.trim() && !row.license_plate.trim()) continue;
        const payload = {
          model: row.model.trim(),
          license_plate: row.license_plate.trim().toUpperCase(),
          wheelchair_accessible: row.wheelchair_accessible,
          is_active: row.is_active,
          insurance_company_name: row.insurance_company_name?.trim() || null,
        };
        if (row.year && String(row.year).trim()) payload.year = parseInt(row.year);
        if (row.seats && String(row.seats).trim()) payload.seats = parseInt(row.seats);
        payload.inspection_expires_at = row.inspection_expires_at || null;
        payload.tachograph_expires_at = row.tachograph_expires_at || null;
        try {
          await createVehicle(payload);
        } catch (err) {
          errors.push(err);
        }
      }

      await loadVehicles();

      if (errors.length > 0) {
        throw errors[0];
      }
    },
  }), [rowEdits, newRows, deletedIds]);

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Chargement des véhicules…</p>
      </div>
    );
  }

  const visibleVehicles = vehicles.filter((v) => !deletedIds.includes(v.id));

  return (
    <div className={`${styles.settingsForm} ${vehicleStyles.settingsFormBlock}`}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div className={styles.cardIcon}><FiTruck size={16} /></div>
          <div className={styles.cardHeaderText}>
            <h3 className={styles.cardTitle}>Flotte de vehicules</h3>
            <p className={styles.cardHint}>
              {vehicles.length} vehicule{vehicles.length !== 1 ? 's' : ''} enregistre{vehicles.length !== 1 ? 's' : ''}
            </p>
          </div>
          {isEditing && (
            <button
              type="button"
              className={`${styles.button} ${styles.primary}`}
              onClick={handleAddRow}
            >
              <FiPlus size={14} aria-hidden />
              Ajouter
            </button>
          )}
        </div>

        {visibleVehicles.length === 0 && newRows.length === 0 ? (
          <div className={vehicleStyles.emptyState}>
            <div className={vehicleStyles.emptyStateIcon}>
              <FiTruck size={36} aria-hidden />
            </div>
            <p className={vehicleStyles.emptyStateText}>Aucun vehicule enregistre</p>
            <p className={vehicleStyles.emptyStateHint}>Ajoutez votre premier vehicule pour commencer</p>
          </div>
        ) : (
          <div className={vehicleStyles.tableContainer}>
            <table className={vehicleStyles.fleetTable}>
              <thead>
                <tr>
                  <th>Vehicule</th>
                  <th>Plaque</th>
                  <th>Annee</th>
                  <th>Places</th>
                  <th>FAH</th>
                  <th>Statut</th>
                  <th>Expertise</th>
                  <th>Assureur</th>
                  <th>Tachygraphe</th>
                  {isEditing && <th className={vehicleStyles.actionsCell}>Actions</th>}
                </tr>
              </thead>
              <tbody>
                {visibleVehicles.map((vehicle) => {
                  const edits = rowEdits[vehicle.id];
                  return (
                    <tr key={vehicle.id}>
                      {isEditing && edits ? (
                        <>
                          <td>
                            <input
                              type="text"
                              value={edits.model}
                              onChange={(e) => updateField(vehicle.id, 'model', e.target.value)}
                              className={vehicleStyles.inlineInput}
                              placeholder="Modèle"
                            />
                          </td>
                          <td>
                            <input
                              type="text"
                              value={edits.license_plate}
                              onChange={(e) => updateField(vehicle.id, 'license_plate', e.target.value)}
                              className={`${vehicleStyles.inlineInput} ${vehicleStyles.inlineInputPlate}`}
                              placeholder="GE 123456"
                            />
                          </td>
                          <td>
                            <input
                              type="number"
                              value={edits.year}
                              onChange={(e) => updateField(vehicle.id, 'year', e.target.value)}
                              className={vehicleStyles.inlineInputSmall}
                              placeholder="2024"
                              min="1950"
                              max="2100"
                            />
                          </td>
                          <td>
                            <input
                              type="number"
                              value={edits.seats}
                              onChange={(e) => updateField(vehicle.id, 'seats', e.target.value)}
                              className={vehicleStyles.inlineInputSmall}
                              placeholder="5"
                              min="0"
                            />
                          </td>
                          <td className={vehicleStyles.inlineToggleCell}>
                            <label className={vehicleStyles.inlineToggleWrap}>
                              <input
                                type="checkbox"
                                checked={edits.wheelchair_accessible}
                                onChange={(e) => updateField(vehicle.id, 'wheelchair_accessible', e.target.checked)}
                              />
                              <span className={vehicleStyles.inlineSlider} />
                            </label>
                          </td>
                          <td className={vehicleStyles.inlineToggleCell}>
                            <label className={vehicleStyles.inlineToggleWrap}>
                              <input
                                type="checkbox"
                                checked={edits.is_active}
                                onChange={(e) => updateField(vehicle.id, 'is_active', e.target.checked)}
                              />
                              <span className={vehicleStyles.inlineSlider} />
                            </label>
                          </td>
                          <td>
                            <InlineDatePicker
                              value={edits.inspection_expires_at || ''}
                              onChange={(v) => updateField(vehicle.id, 'inspection_expires_at', v || null)}
                              placeholder="CT"
                            />
                          </td>
                          <td>
                            <input
                              type="text"
                              value={edits.insurance_company_name}
                              onChange={(e) => updateField(vehicle.id, 'insurance_company_name', e.target.value)}
                              className={vehicleStyles.inlineInput}
                              placeholder="Assureur"
                            />
                          </td>
                          <td>
                            <InlineDatePicker
                              value={edits.tachograph_expires_at || ''}
                              onChange={(v) => updateField(vehicle.id, 'tachograph_expires_at', v || null)}
                              placeholder="Tachy"
                            />
                          </td>
                          <td className={vehicleStyles.actionsCell}>
                            <button
                              type="button"
                              className={`${vehicleStyles.actionBtn} ${vehicleStyles.actionBtnDelete}`}
                              onClick={() => handleMarkDelete(vehicle.id)}
                              title="Supprimer"
                            >
                              <FiTrash2 size={15} aria-hidden />
                            </button>
                          </td>
                        </>
                      ) : (
                        <>
                          <td><strong>{vehicle.model || '\u2014'}</strong></td>
                          <td className={vehicleStyles.licensePlateCell}>{vehicle.license_plate || '\u2014'}</td>
                          <td>{vehicle.year || '\u2014'}</td>
                          <td>{vehicle.seats || '\u2014'}</td>
                          <td>
                            {vehicle.wheelchair_accessible ? (
                              <span className={vehicleStyles.fahBadgeYes}><FiCheck size={13} /> Oui</span>
                            ) : (
                              <span className={vehicleStyles.fahBadgeNo}><FiX size={13} /> Non</span>
                            )}
                          </td>
                          <td>
                            <span className={`${vehicleStyles.statusBadge} ${
                              vehicle.is_active ? vehicleStyles.statusBadgeActive : vehicleStyles.statusBadgeInactive
                            }`}>
                              {vehicle.is_active ? 'Actif' : 'Inactif'}
                            </span>
                          </td>
                          {(() => { const b = getControlBadge(vehicle.inspection_expires_at, 1); return <td><span className={`${vehicleStyles.dateBadge} ${b.cls}`}>{b.label}</span></td>; })()}
                          <td>{vehicle.insurance_company_name || '\u2014'}</td>
                          {(() => { const b = getControlBadge(vehicle.tachograph_expires_at, 2); return <td><span className={`${vehicleStyles.dateBadge} ${b.cls}`}>{b.label}</span></td>; })()}
                        </>
                      )}
                    </tr>
                  );
                })}

                {newRows.map((row) => (
                  <tr key={row._tempId} className={vehicleStyles.newRow}>
                    <td>
                      <input
                        type="text"
                        value={row.model}
                        onChange={(e) => updateNewRow(row._tempId, 'model', e.target.value)}
                        className={vehicleStyles.inlineInput}
                        placeholder="Modèle *"
                      />
                    </td>
                    <td>
                      <input
                        type="text"
                        value={row.license_plate}
                        onChange={(e) => updateNewRow(row._tempId, 'license_plate', e.target.value)}
                        className={`${vehicleStyles.inlineInput} ${vehicleStyles.inlineInputPlate}`}
                        placeholder="GE 123456 *"
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        value={row.year}
                        onChange={(e) => updateNewRow(row._tempId, 'year', e.target.value)}
                        className={vehicleStyles.inlineInputSmall}
                        placeholder="2024"
                        min="1950"
                        max="2100"
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        value={row.seats}
                        onChange={(e) => updateNewRow(row._tempId, 'seats', e.target.value)}
                        className={vehicleStyles.inlineInputSmall}
                        placeholder="5"
                        min="0"
                      />
                    </td>
                    <td className={vehicleStyles.inlineToggleCell}>
                      <label className={vehicleStyles.inlineToggleWrap}>
                        <input
                          type="checkbox"
                          checked={row.wheelchair_accessible}
                          onChange={(e) => updateNewRow(row._tempId, 'wheelchair_accessible', e.target.checked)}
                        />
                        <span className={vehicleStyles.inlineSlider} />
                      </label>
                    </td>
                    <td className={vehicleStyles.inlineToggleCell}>
                      <label className={vehicleStyles.inlineToggleWrap}>
                        <input
                          type="checkbox"
                          checked={row.is_active}
                          onChange={(e) => updateNewRow(row._tempId, 'is_active', e.target.checked)}
                        />
                        <span className={vehicleStyles.inlineSlider} />
                      </label>
                    </td>
                    <td>
                      <InlineDatePicker
                        value={row.inspection_expires_at || ''}
                        onChange={(v) => updateNewRow(row._tempId, 'inspection_expires_at', v || null)}
                        placeholder="CT"
                      />
                    </td>
                    <td>
                      <input
                        type="text"
                        value={row.insurance_company_name}
                        onChange={(e) => updateNewRow(row._tempId, 'insurance_company_name', e.target.value)}
                        className={vehicleStyles.inlineInput}
                        placeholder="Assureur"
                      />
                    </td>
                    <td>
                      <InlineDatePicker
                        value={row.tachograph_expires_at || ''}
                        onChange={(v) => updateNewRow(row._tempId, 'tachograph_expires_at', v || null)}
                        placeholder="Tachy"
                      />
                    </td>
                    <td className={vehicleStyles.actionsCell}>
                      <button
                        type="button"
                        className={`${vehicleStyles.actionBtn} ${vehicleStyles.actionBtnDelete}`}
                        onClick={() => handleRemoveNewRow(row._tempId)}
                        title="Retirer"
                      >
                        <FiX size={15} aria-hidden />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
});

VehiclesTab.displayName = 'VehiclesTab';

export default VehiclesTab;
