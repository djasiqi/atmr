import React, { useState, useEffect } from 'react';
import { invoiceService } from '../../../../../services/invoiceService';
import styles from './ReservationSelector.module.css';

const ReservationSelector = ({ 
  companyId, 
  clientId,
  clientName,
  period, 
  billToType,
  onSelectionChange 
}) => {
  const [reservations, setReservations] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [filter, setFilter] = useState('clinic'); // Filtre par défaut sur 'clinic' pour facturation tierce
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasAutoSelected, setHasAutoSelected] = useState(false);

  // Charger les réservations non facturées
  useEffect(() => {
    const loadReservations = async () => {
      if (!companyId || !clientId || !period.year || !period.month) return;
      
      try {
        setLoading(true);
        setError(null);
        
        const data = await invoiceService.fetchUnbilledReservations(
          companyId,
          clientId,
          {
            year: period.year,
            month: period.month,
            billed_to_type: filter !== 'all' ? filter : undefined
          }
        );
        
        setReservations(data.reservations || []);
      } catch (err) {
        console.error('Erreur chargement réservations:', err);
        setError('Erreur lors du chargement des transports');
        setReservations([]);
      } finally {
        setLoading(false);
      }
    };

    loadReservations();
  }, [companyId, clientId, period, filter]);

  // Auto-sélectionner les transports qui matchent le type de facturation (une seule fois)
  useEffect(() => {
    if (billToType && reservations.length > 0 && !hasAutoSelected) {
      const matching = reservations
        .filter(r => r.billed_to_type === billToType)
        .map(r => r.id);
      setSelectedIds(matching);
      setHasAutoSelected(true);
    }
  }, [reservations, billToType, hasAutoSelected]);

  // Notifier le parent des changements de sélection
  useEffect(() => {
    if (onSelectionChange) {
      const selected = reservations.filter(r => selectedIds.includes(r.id));
      onSelectionChange(selected);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIds]);

  const handleToggle = (reservationId) => {
    setSelectedIds(prev =>
      prev.includes(reservationId)
        ? prev.filter(id => id !== reservationId)
        : [...prev, reservationId]
    );
  };

  const handleSelectAll = () => {
    const allIds = reservations.map(r => r.id);
    setSelectedIds(allIds);
  };

  const handleDeselectAll = () => {
    setSelectedIds([]);
  };

  const selectedReservations = reservations.filter(r => selectedIds.includes(r.id));
  const totalSelected = selectedReservations.reduce((sum, r) => sum + (r.amount || 0), 0);

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      return new Date(dateString).toLocaleDateString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
      });
    } catch {
      return '-';
    }
  };

  const getBillingTypeLabel = (type) => {
    const labels = {
      patient: '👤 Patient',
      clinic: '🏥 Clinique',
      insurance: '🏢 Assurance'
    };
    return labels[type] || type;
  };

  const getBillingTypeClass = (type) => {
    return type || 'patient';
  };

  if (loading) {
    return (
      <div className={styles.loading}>
        Chargement des transports...
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.error}>
        {error}
      </div>
    );
  }

  if (reservations.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>🚗</div>
        <p>Aucun transport non facturé pour cette période</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h4 className={styles.clientName}>{clientName}</h4>
        <div className={styles.filterButtons}>
          <button
            type="button"
            className={`${styles.filterBtn} ${filter === 'all' ? styles.active : ''}`}
            onClick={() => setFilter('all')}
          >
            Tous ({reservations.length})
          </button>
          <button
            type="button"
            className={`${styles.filterBtn} ${filter === 'clinic' ? styles.active : ''}`}
            onClick={() => setFilter('clinic')}
          >
            🏥 Clinique
          </button>
          <button
            type="button"
            className={`${styles.filterBtn} ${filter === 'patient' ? styles.active : ''}`}
            onClick={() => setFilter('patient')}
          >
            👤 Patient
          </button>
        </div>
      </div>

      <div className={styles.actions}>
        <button type="button" onClick={handleSelectAll} className={styles.actionBtn}>
          Tout sélectionner
        </button>
        <button type="button" onClick={handleDeselectAll} className={styles.actionBtn}>
          Tout désélectionner
        </button>
      </div>

      <div className={styles.reservationsList}>
        {reservations.map(reservation => (
          <label
            key={reservation.id}
            className={`${styles.reservationItem} ${
              selectedIds.includes(reservation.id) ? styles.selected : ''
            }`}
          >
            <input
              type="checkbox"
              checked={selectedIds.includes(reservation.id)}
              onChange={() => handleToggle(reservation.id)}
              className={styles.checkbox}
            />
            
            <div className={styles.reservationContent}>
              <div className={styles.reservationHeader}>
                <span className={styles.date}>
                  {formatDate(reservation.date)}
                </span>
                <span className={styles.amount}>
                  {(reservation.amount || 0).toFixed(2)} CHF
                </span>
              </div>
              
              <div className={styles.route}>
                <span className={styles.location}>{reservation.pickup_location}</span>
                <span className={styles.arrow}>→</span>
                <span className={styles.location}>{reservation.dropoff_location}</span>
              </div>
              
              <div className={styles.reservationFooter}>
                <span className={`${styles.badge} ${styles[getBillingTypeClass(reservation.billed_to_type)]}`}>
                  {getBillingTypeLabel(reservation.billed_to_type)}
                </span>
                {reservation.is_return && (
                  <span className={styles.returnBadge}>↩ Retour</span>
                )}
                {reservation.is_urgent && (
                  <span className={styles.urgentBadge}>⚡ Urgent</span>
                )}
                {reservation.medical_facility && (
                  <span className={styles.medicalBadge}>
                    🏥 {reservation.medical_facility}
                  </span>
                )}
              </div>
            </div>
          </label>
        ))}
      </div>

      <div className={styles.summary}>
        <div className={styles.summaryItem}>
          <strong>{selectedIds.length}</strong> transport(s) sélectionné(s)
        </div>
        <div className={styles.summaryItem}>
          <strong>{totalSelected.toFixed(2)} CHF</strong>
        </div>
      </div>
    </div>
  );
};

export default ReservationSelector;

