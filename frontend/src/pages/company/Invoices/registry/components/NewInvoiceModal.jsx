import React, { useState, useEffect, useCallback } from 'react';
import styles from './NewInvoiceModal.module.css';
import { generateInvoice, invoiceService } from '../../../../../services/invoiceService';
import { fetchCompanyClients } from '../../../../../services/companyService';
import ReservationSelector from './ReservationSelector';

const NewInvoiceModal = ({ open, onClose, onInvoiceGenerated, companyId }) => {
  const [billingType, setBillingType] = useState('direct'); // 'direct' ou 'third_party'
  const [formData, setFormData] = useState({
    client_id: '',
    client_ids: [],
    bill_to_client_id: '',
    period_year: new Date().getFullYear(),
    period_month: new Date().getMonth() + 1,
  });
  const [clients, setClients] = useState([]);
  const [institutions, setInstitutions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  
  // NOUVEAU: Gestion des sélections de réservations par client
  const [selectedReservations, setSelectedReservations] = useState({});  // { client_id: [reservation_objects] }
  const [showReservationSelection, setShowReservationSelection] = useState(false);

  // Charger la liste des clients et institutions
  useEffect(() => {
    const loadData = async () => {
      if (!companyId) return;
      
      try {
        setLoading(true);
        
        // Charger les clients
        const clientsData = await fetchCompanyClients();
        // Filtrer pour ne garder que les clients non-institutions pour la sélection patient
        const regularClients = clientsData.filter(c => !c.is_institution);
        setClients(regularClients);
        
        // Charger les institutions
        const institutionsData = await invoiceService.fetchInstitutions(companyId);
        setInstitutions(institutionsData.institutions || []);
        
      } catch (err) {
        console.error('Erreur lors du chargement des données:', err);
        setError('Erreur lors du chargement des données');
      } finally {
        setLoading(false);
      }
    };

    if (open) {
      loadData();
    }
  }, [companyId, open]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.includes('year') || name.includes('month') ? parseInt(value) : value
    }));
  };

  const handleClientToggle = (clientId) => {
    setFormData(prev => {
      const isSelected = prev.client_ids.includes(clientId);
      const newClientIds = isSelected
        ? prev.client_ids.filter(id => id !== clientId)
        : [...prev.client_ids, clientId];
      
      // Si on désélectionne un client, supprimer aussi ses réservations sélectionnées
      if (isSelected) {
        setSelectedReservations(prevReservations => {
          if (!prevReservations) return {};
          const { [clientId]: removed, ...rest } = prevReservations;
          return rest;
        });
      }
      
      return {
        ...prev,
        client_ids: newClientIds
      };
    });
  };

  // IMPORTANT: Utiliser useCallback pour éviter les re-renders infinis
  const handleReservationSelectionChange = useCallback((clientId, reservations) => {
    setSelectedReservations(prev => {
      // Vérifier que prev existe, sinon initialiser à {}
      const current = prev || {};
      
      // Ne mettre à jour que si les réservations ont changé
      const prevIds = (current[clientId] || []).map(r => r?.id || r).sort().join(',');
      const newIds = (reservations || []).map(r => r?.id || r).sort().join(',');
      
      if (prevIds === newIds) {
        return current; // Pas de changement, retourner le même objet
      }
      
      return {
        ...current,
        [clientId]: reservations || []
      };
    });
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation en fonction du type de facturation
    if (billingType === 'direct' && !formData.client_id) {
      setError('Veuillez sélectionner un client');
      return;
    }
    
    if (billingType === 'third_party') {
      if (formData.client_ids.length === 0) {
        setError('Veuillez sélectionner au moins un patient');
        return;
      }
      if (!formData.bill_to_client_id) {
        setError('Veuillez sélectionner une institution payeuse');
        return;
      }
    }

    try {
      setLoading(true);
      setError(null);
      setSuccessMessage(null);
      
      let result;
      
      if (billingType === 'direct') {
        // Facturation directe
        const clientId = parseInt(formData.client_id);
        const reservs = selectedReservations?.[clientId];
        const reservationIds = Array.isArray(reservs) ? reservs.map(r => r?.id || r) : undefined;
        
        result = await generateInvoice(companyId, {
          client_id: clientId,
          period_year: formData.period_year,
          period_month: formData.period_month,
          reservation_ids: reservationIds  // NOUVEAU: Support sélection manuelle
        });
        
        // Ouvrir le PDF dans un nouvel onglet
        if (result.pdf_url) {
          window.open(result.pdf_url, '_blank');
        }
        
        onInvoiceGenerated(result);
        
      } else {
        // Facturation tierce (consolidée)
        // NOUVEAU: Préparer le mapping des réservations par client
        const clientReservations = {};
        formData.client_ids.forEach(clientId => {
          const reservs = selectedReservations?.[clientId];
          if (reservs && Array.isArray(reservs) && reservs.length > 0) {
            clientReservations[clientId] = reservs.map(r => r?.id || r);
          }
        });
        
        result = await invoiceService.generateConsolidatedInvoice(companyId, {
          client_ids: formData.client_ids.map(id => parseInt(id)),
          bill_to_client_id: parseInt(formData.bill_to_client_id),
          period_year: formData.period_year,
          period_month: formData.period_month,
          client_reservations: Object.keys(clientReservations).length > 0 ? clientReservations : undefined  // NOUVEAU
        });
        
        if (result.invoices && result.invoices.length > 0) {
          setSuccessMessage(
            `${result.success_count} facture(s) générée(s) avec succès${result.error_count > 0 ? `, ${result.error_count} erreur(s)` : ''}`
          );
          
          // Ouvrir les PDFs dans de nouveaux onglets
          result.invoices.forEach(inv => {
            if (inv.pdf_url) {
              window.open(inv.pdf_url, '_blank');
            }
          });
          
          // Notifier le parent pour chaque facture
          result.invoices.forEach(inv => onInvoiceGenerated(inv));
        }
        
        if (result.errors && result.errors.length > 0) {
          const errorMessages = result.errors.map(e => `Client ${e.client_id}: ${e.error}`).join('\n');
          setError(`Erreurs:\n${errorMessages}`);
        }
      }
      
      // Fermer le modal si tout s'est bien passé et pas d'erreurs
      if (!result.errors || result.errors.length === 0) {
        setTimeout(() => {
          onClose();
        }, 2000);
      }
      
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Erreur lors de la génération de la facture');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setError(null);
    setSuccessMessage(null);
    setBillingType('direct');
    setSelectedReservations({});
    setShowReservationSelection(false);
    setFormData({
      client_id: '',
      client_ids: [],
      bill_to_client_id: '',
      period_year: new Date().getFullYear(),
      period_month: new Date().getMonth() + 1,
    });
    onClose();
  };

  if (!open) return null;

  const months = [
    { value: 1, label: 'Janvier' },
    { value: 2, label: 'Février' },
    { value: 3, label: 'Mars' },
    { value: 4, label: 'Avril' },
    { value: 5, label: 'Mai' },
    { value: 6, label: 'Juin' },
    { value: 7, label: 'Juillet' },
    { value: 8, label: 'Août' },
    { value: 9, label: 'Septembre' },
    { value: 10, label: 'Octobre' },
    { value: 11, label: 'Novembre' },
    { value: 12, label: 'Décembre' },
  ];

  const years = Array.from({ length: 3 }, (_, i) => new Date().getFullYear() - i);

  return (
    <div className={styles.modalOverlay}>
      <div className={styles.modal}>
        <div className={styles.modalHeader}>
          <h2>Nouvelle facture</h2>
          <button className={styles.closeBtn} onClick={handleClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && (
            <div className={styles.error}>
              {error}
            </div>
          )}

          {successMessage && (
            <div className={styles.success}>
              {successMessage}
            </div>
          )}

          {/* Type de facturation */}
          <div className={styles.formGroup}>
            <label className={styles.label}>Type de facturation</label>
            <div className={styles.radioGroup}>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  value="direct"
                  checked={billingType === 'direct'}
                  onChange={(e) => setBillingType(e.target.value)}
                  disabled={loading}
                />
                Facturation directe au client
              </label>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  value="third_party"
                  checked={billingType === 'third_party'}
                  onChange={(e) => setBillingType(e.target.value)}
                  disabled={loading}
                />
                Facturation tierce (clinique)
              </label>
            </div>
          </div>

          {/* Facturation directe */}
          {billingType === 'direct' && (
            <>
              <div className={styles.formGroup}>
                <label htmlFor="client_id" className={styles.label}>
                  Client *
                </label>
                <select
                  id="client_id"
                  name="client_id"
                  value={formData.client_id}
                  onChange={handleInputChange}
                  className={styles.select}
                  required
                  disabled={loading}
                >
                  <option value="">Sélectionner un client</option>
                  {clients.map((client) => (
                    <option key={client.id} value={client.id}>
                      {`${client.first_name || ''} ${client.last_name || ''}`.trim() || client.username}
                    </option>
                  ))}
                </select>
              </div>

              {/* Sélection des transports pour facturation directe */}
              {formData.client_id && (
                <div className={styles.formGroup}>
                  <div className={styles.sectionHeader}>
                    <label className={styles.label}>
                      Transports à facturer
                    </label>
                    <button
                      type="button"
                      className={styles.toggleBtn}
                      onClick={() => setShowReservationSelection(!showReservationSelection)}
                    >
                      {showReservationSelection ? '▼ Masquer' : '▶ Sélectionner'}
                    </button>
                  </div>
                  
                  {showReservationSelection && (
                    <ReservationSelector
                      companyId={companyId}
                      clientId={parseInt(formData.client_id)}
                      clientName={clients.find(c => c.id === parseInt(formData.client_id))?.full_name || ''}
                      period={{ year: formData.period_year, month: formData.period_month }}
                      billToType="patient"
                      onSelectionChange={(reservations) => 
                        handleReservationSelectionChange(parseInt(formData.client_id), reservations)
                      }
                    />
                  )}
                  
                  {selectedReservations?.[parseInt(formData.client_id)]?.length > 0 && (
                    <small className={styles.hint}>
                      {selectedReservations[parseInt(formData.client_id)].length} transport(s) sélectionné(s)
                    </small>
                  )}
                </div>
              )}
            </>
          )}

          {/* Facturation tierce */}
          {billingType === 'third_party' && (
            <>
              <div className={styles.formGroup}>
                <label htmlFor="bill_to_client_id" className={styles.label}>
                  Institution payeuse *
                </label>
                <select
                  id="bill_to_client_id"
                  name="bill_to_client_id"
                  value={formData.bill_to_client_id}
                  onChange={handleInputChange}
                  className={styles.select}
                  required
                  disabled={loading}
                >
                  <option value="">Sélectionner une institution</option>
                  {institutions.map((inst) => (
                    <option key={inst.id} value={inst.id}>
                      {inst.institution_name}
                    </option>
                  ))}
                </select>
                {institutions.length === 0 && (
                  <small className={styles.hint}>
                    Aucune institution disponible. Créez d'abord des clients institutions.
                  </small>
                )}
              </div>

              <div className={styles.formGroup}>
                <label className={styles.label}>
                  Sélection des patients
                </label>
                
                {/* Liste simplifiée pour sélectionner les patients */}
                <div className={styles.clientsList}>
                  {clients.map((client) => (
                    <label key={client.id} className={styles.checkboxLabel}>
                      <input
                        type="checkbox"
                        checked={formData.client_ids.includes(client.id)}
                        onChange={() => handleClientToggle(client.id)}
                        disabled={loading}
                      />
                      {`${client.first_name || ''} ${client.last_name || ''}`.trim() || client.username}
                    </label>
                  ))}
                </div>
              </div>

              {/* Sélection des transports pour chaque patient sélectionné */}
              {formData.client_ids.length > 0 && formData.period_year && formData.period_month && (
                <div className={styles.formGroup}>
                  <label className={styles.label}>
                    Transports à facturer
                  </label>
                  
                  <div className={styles.patientsWithReservations}>
                    {formData.client_ids.map((clientId) => {
                      const client = clients.find(c => c.id === clientId);
                      if (!client) return null;
                      
                      const reservationsCount = selectedReservations?.[clientId]?.length || 0;
                      
                      return (
                        <div key={clientId} className={styles.patientSection}>
                          <div className={styles.patientSectionHeader}>
                            <h4 className={styles.patientName}>
                              {`${client.first_name || ''} ${client.last_name || ''}`.trim() || client.username}
                            </h4>
                            {reservationsCount > 0 && (
                              <span className={styles.reservationCount}>
                                {reservationsCount} transport(s)
                              </span>
                            )}
                          </div>
                          
                          <div className={styles.patientReservations}>
                            <ReservationSelector
                              key={`${clientId}-${formData.period_year}-${formData.period_month}`}
                              companyId={companyId}
                              clientId={clientId}
                              clientName={client.full_name || `${client.first_name} ${client.last_name}`}
                              period={{ year: formData.period_year, month: formData.period_month }}
                              billToType="clinic"
                              onSelectionChange={handleReservationSelectionChange}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              
              <small className={styles.hint}>
                {formData.client_ids.length} patient(s) sélectionné(s) • {' '}
                {Object.values(selectedReservations || {}).reduce((sum, res) => sum + (res?.length || 0), 0)} transport(s) au total
              </small>
            </>
          )}

          {/* Période */}
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="period_year" className={styles.label}>
                Année
              </label>
              <select
                id="period_year"
                name="period_year"
                value={formData.period_year}
                onChange={handleInputChange}
                className={styles.select}
                disabled={loading}
              >
                {years.map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="period_month" className={styles.label}>
                Mois
              </label>
              <select
                id="period_month"
                name="period_month"
                value={formData.period_month}
                onChange={handleInputChange}
                className={styles.select}
                disabled={loading}
              >
                {months.map((month) => (
                  <option key={month.value} value={month.value}>
                    {month.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className={styles.modalActions}>
            <button
              type="button"
              onClick={handleClose}
              className={styles.cancelBtn}
              disabled={loading}
            >
              Annuler
            </button>
            <button
              type="submit"
              className={styles.submitBtn}
              disabled={
                loading ||
                (billingType === 'direct' && !formData.client_id) ||
                (billingType === 'third_party' && (formData.client_ids.length === 0 || !formData.bill_to_client_id))
              }
            >
              {loading ? 'Génération...' : 'Générer la facture'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NewInvoiceModal;
