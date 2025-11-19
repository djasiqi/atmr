import React, { useState, useEffect, useCallback, useMemo } from 'react';
import styles from './NewInvoiceModal.module.css';
import { generateInvoice, invoiceService } from '../../../../../services/invoiceService';
import ReservationSelector from './ReservationSelector';

const NewInvoiceModal = ({ open, onClose, onInvoiceGenerated, companyId, initialDraft = null }) => {
  const [billingType, setBillingType] = useState('direct'); // 'direct' ou 'third_party'
  const [formData, setFormData] = useState({
    client_id: '',
    client_ids: [],
    bill_to_client_id: '',
    period_year: new Date().getFullYear(),
    period_month: new Date().getMonth() + 1,
  });
  const [clients, setClients] = useState([]);
  const [clientCache, setClientCache] = useState({});
  const [clientSearch, setClientSearch] = useState('');
  const [clientsLoading, setClientsLoading] = useState(false);
  const [clientsError, setClientsError] = useState(null);
  const [institutions, setInstitutions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // NOUVEAU: Gestion des sélections de réservations par client
  const [selectedReservations, setSelectedReservations] = useState({}); // { client_id: [reservation_objects] }
  const [showReservationSelection, setShowReservationSelection] = useState(false);
  const [overrides, setOverrides] = useState({});
  const [preselectedReservations, setPreselectedReservations] = useState({});
  useEffect(() => {
    if (!open) return;

    if (initialDraft) {
      const billing = initialDraft.billing_type === 'third_party' ? 'third_party' : 'direct';
      setBillingType(billing);

      setFormData({
        client_id: initialDraft.client_id ? String(initialDraft.client_id) : '',
        client_ids:
          billing === 'third_party' && Array.isArray(initialDraft.client_ids)
            ? initialDraft.client_ids
            : [],
        bill_to_client_id: initialDraft.bill_to_client_id
          ? String(initialDraft.bill_to_client_id)
          : '',
        period_year: initialDraft.period_year ?? new Date().getFullYear(),
        period_month: initialDraft.period_month ?? new Date().getMonth() + 1,
      });

      setOverrides(initialDraft.overrides || {});
      if (
        Array.isArray(initialDraft.reservation_ids) &&
        initialDraft.client_id &&
        initialDraft.reservation_ids.length > 0
      ) {
        setPreselectedReservations({
          [initialDraft.client_id]: initialDraft.reservation_ids.map((id) => Number(id)),
        });
      } else {
        setPreselectedReservations({});
      }

      if (initialDraft.client) {
        setClientCache((prev) => ({ ...prev, [initialDraft.client.id]: initialDraft.client }));
        setClients((prev) => {
          if (prev.some((c) => c.id === initialDraft.client.id)) {
            return prev;
          }
          return [...prev, initialDraft.client];
        });
      }

      setSelectedReservations({});
      setClientSearch('');
      setShowReservationSelection(true);
      return;
    }

    // Réinitialiser les champs pour une création manuelle
    setBillingType('direct');
    setFormData({
      client_id: '',
      client_ids: [],
      bill_to_client_id: '',
      period_year: new Date().getFullYear(),
      period_month: new Date().getMonth() + 1,
    });
    setOverrides({});
    setSelectedReservations({});
    setPreselectedReservations({});
    setShowReservationSelection(false);
    setClientSearch('');
  }, [open, initialDraft]);
  const [vatConfig, setVatConfig] = useState({
    applicable: false,
    defaultRate: 0,
    label: '',
    number: '',
  });

  // Charger la liste des institutions à l'ouverture du modal
  useEffect(() => {
    if (!open || !companyId) return;

    let isMounted = true;

    const loadInstitutions = async () => {
      try {
        setLoading(true);
        const institutionsData = await invoiceService.fetchInstitutions(companyId);
        if (!isMounted) return;
        setInstitutions(institutionsData.institutions || []);
      } catch (err) {
        console.error('Erreur lors du chargement des institutions:', err);
        if (isMounted) {
          setError('Erreur lors du chargement des institutions');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    loadInstitutions();

    return () => {
      isMounted = false;
    };
  }, [companyId, open]);

  useEffect(() => {
    if (!open || !companyId) return;
    let cancelled = false;

    const loadBillingSettings = async () => {
      try {
        const settings = await invoiceService.fetchBillingSettings(companyId);
        if (cancelled || !settings) return;
        setVatConfig({
          applicable: Boolean(settings.vat_applicable),
          defaultRate:
            settings.vat_rate !== undefined && settings.vat_rate !== null
              ? Number(settings.vat_rate)
              : 0,
          label: settings.vat_label ?? '',
          number: settings.vat_number ?? '',
        });
      } catch (err) {
        console.warn('Erreur chargement paramètres TVA:', err);
        if (!cancelled) {
          setVatConfig((prev) => ({
            ...prev,
            applicable: false,
            defaultRate: 0,
          }));
        }
      }
    };

    loadBillingSettings();

    return () => {
      cancelled = true;
    };
  }, [companyId, open]);

  // Charger les clients éligibles (trajets non facturés) avec recherche
  useEffect(() => {
    if (!open || !companyId) return;

    let cancelled = false;

    const fetchClients = async () => {
      try {
        setClientsLoading(true);
        setClientsError(null);
        const query = clientSearch.trim();

        const response = await invoiceService.fetchEligibleClients(companyId, {
          search: query || undefined,
          limit: 120,
        });
        const list = Array.isArray(response?.clients) ? response.clients : [];

        if (!list.length) {
          setClientsError(
            query
              ? 'Aucun client trouvé pour cette recherche.'
              : "Aucun client éligible (courses terminées non facturées) n'a été trouvé pour cette période."
          );
        }

        if (cancelled) return;

        setClients(list);
        setClientCache((prev) => {
          const next = { ...prev };
          list.forEach((client) => {
            if (client && client.id != null) {
              next[client.id] = client;
            }
          });
          return next;
        });
      } catch (err) {
        console.error('Erreur lors du chargement des clients éligibles:', err);
        if (!cancelled) {
          setClients([]);
          setClientsError(
            'Impossible de charger les clients à facturer. Vérifiez que votre backend est à jour.'
          );
        }
      } finally {
        if (!cancelled) {
          setClientsLoading(false);
        }
      }
    };

    const timer = setTimeout(fetchClients, 250);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [companyId, open, clientSearch]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name.includes('year') || name.includes('month') ? parseInt(value) : value,
    }));
  };

  const handleClientToggle = (clientId) => {
    setFormData((prev) => {
      const isSelected = prev.client_ids.includes(clientId);
      const newClientIds = isSelected
        ? prev.client_ids.filter((id) => id !== clientId)
        : [...prev.client_ids, clientId];

      // Si on désélectionne un client, supprimer aussi ses réservations sélectionnées
      if (isSelected) {
        setSelectedReservations((prevReservations) => {
          if (!prevReservations) return {};
          const { [clientId]: _removed, ...rest } = prevReservations;
          return rest;
        });
      }

      return {
        ...prev,
        client_ids: newClientIds,
      };
    });
  };

  // IMPORTANT: Utiliser useCallback pour éviter les re-renders infinis
  const handleReservationSelectionChange = useCallback((clientId, reservations) => {
    setSelectedReservations((prev) => {
      // Vérifier que prev existe, sinon initialiser à {}
      const current = prev || {};

      // Ne mettre à jour que si les réservations ont changé
      const prevIds = (current[clientId] || [])
        .map((r) => r?.id || r)
        .sort()
        .join(',');
      const newIds = (reservations || [])
        .map((r) => r?.id || r)
        .sort()
        .join(',');

      if (prevIds === newIds) {
        return current; // Pas de changement, retourner le même objet
      }

      return {
        ...current,
        [clientId]: reservations || [],
      };
    });
  }, []);

  const handleOverrideChange = useCallback((reservationId, patch) => {
    const key = String(reservationId);
    setOverrides((prev) => {
      const current = prev[key] ? { ...prev[key] } : {};
      let changed = false;

      Object.entries(patch).forEach(([field, value]) => {
        if (value === null || value === undefined || value === '') {
          if (field in current) {
            delete current[field];
            changed = true;
          }
        } else if (current[field] !== value) {
          current[field] = value;
          changed = true;
        }
      });

      const next = { ...prev };
      if (Object.keys(current).length === 0) {
        if (next[key]) {
          delete next[key];
          changed = true;
        }
      } else {
        next[key] = current;
      }

      return changed ? next : prev;
    });
  }, []);

  const selectedClientIds = useMemo(() => {
    const ids = new Set();
    if (formData.client_id) {
      const parsed = parseInt(formData.client_id, 10);
      if (!Number.isNaN(parsed)) ids.add(parsed);
    }
    formData.client_ids.forEach((value) => {
      const parsed = parseInt(value, 10);
      if (!Number.isNaN(parsed)) ids.add(parsed);
    });
    return Array.from(ids);
  }, [formData.client_id, formData.client_ids]);

  const selectedClients = useMemo(() => {
    return selectedClientIds.map((id) => clientCache[id]).filter(Boolean);
  }, [selectedClientIds, clientCache]);

  const allClients = useMemo(() => {
    const seen = new Set();
    const ordered = [];

    selectedClients.forEach((client) => {
      if (client && !seen.has(client.id)) {
        seen.add(client.id);
        ordered.push(client);
      }
    });

    clients.forEach((client) => {
      if (client && !seen.has(client.id)) {
        seen.add(client.id);
        ordered.push(client);
      }
    });

    return ordered;
  }, [clients, selectedClients]);

  useEffect(() => {
    const hasPendingPreselection = Object.values(preselectedReservations).some(
      (ids) => Array.isArray(ids) && ids.length > 0
    );
    if (
      hasPendingPreselection &&
      (!selectedReservations || Object.keys(selectedReservations || {}).length === 0)
    ) {
      return;
    }

    const activeIds = new Set();
    Object.values(selectedReservations).forEach((list) => {
      (list || []).forEach((reservation) => {
        if (reservation?.id != null) {
          activeIds.add(String(reservation.id));
        }
      });
    });

    setOverrides((prev) => {
      let changed = false;
      const next = { ...prev };
      Object.keys(prev).forEach((key) => {
        if (!activeIds.has(key)) {
          delete next[key];
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [selectedReservations, preselectedReservations]);

  const directClient = useMemo(() => {
    if (!formData.client_id) return null;
    const target = parseInt(formData.client_id, 10);
    if (Number.isNaN(target)) return null;
    return allClients.find((client) => client.id === target) || null;
  }, [allClients, formData.client_id]);

  const computeTotals = useCallback(
    (reservationsList = []) => {
      return reservationsList.reduce(
        (acc, reservation) => {
          const override = overrides[String(reservation?.id)] || {};
          const baseAmount = Number(
            override.amount ?? reservation?.amount ?? reservation?.estimated_amount ?? 0
          );
          const amount = Number.isNaN(baseAmount) ? 0 : baseAmount;
          const vatRate = vatConfig.applicable
            ? Number(
                reservation?.vat_rate ?? reservation?.default_vat_rate ?? vatConfig.defaultRate ?? 0
              )
            : 0;
          const sanitizedRate = Number.isNaN(vatRate) ? 0 : vatRate;
          const vatValue = vatConfig.applicable
            ? Number(((amount * sanitizedRate) / 100).toFixed(2))
            : 0;
          const total = Number((amount + vatValue).toFixed(2));

          acc.base += amount;
          acc.vat += vatValue;
          acc.total += total;
          return acc;
        },
        { base: 0, vat: 0, total: 0 }
      );
    },
    [overrides, vatConfig]
  );

  const activeClientId = formData.client_id ? parseInt(formData.client_id, 10) : null;
  const directSelection = useMemo(() => {
    if (!activeClientId) return [];
    return selectedReservations[activeClientId] || [];
  }, [activeClientId, selectedReservations]);
  const directTotals = useMemo(
    () => computeTotals(directSelection),
    [computeTotals, directSelection]
  );

  const consolidatedSelection = useMemo(
    () => Object.values(selectedReservations).reduce((acc, list) => acc.concat(list || []), []),
    [selectedReservations]
  );
  const consolidatedTotals = useMemo(
    () => computeTotals(consolidatedSelection),
    [computeTotals, consolidatedSelection]
  );

  const formatCurrency = useCallback((value) => `${Number(value || 0).toFixed(2)} CHF`, []);

  const buildOverridesPayload = useCallback(
    (reservationsList = []) => {
      const payload = {};
      reservationsList.forEach((reservation) => {
        if (!reservation || reservation.id == null) return;
        const override = overrides[String(reservation.id)];
        if (!override) return;
        const clean = {};
        if (override.amount !== undefined) {
          const amount = Number(override.amount);
          if (!Number.isNaN(amount)) clean.amount = amount;
        }
        if (override.note) {
          clean.note = override.note;
        }
        if (Object.keys(clean).length > 0) {
          payload[reservation.id] = clean;
        }
      });
      return payload;
    },
    [overrides]
  );

  const formatClientLabel = useCallback((client) => {
    if (!client) return 'Client';
    const name =
      (client.full_name && client.full_name.trim()) ||
      `${client.first_name || ''} ${client.last_name || ''}`.trim() ||
      client.username ||
      `Client #${client.id}`;
    const count = client.unbilled_count ?? 0;
    const suffix = count > 1 ? 's' : '';
    return `${name} • ${count} transport${suffix}`;
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
        const reservs = Array.isArray(selectedReservations?.[clientId])
          ? selectedReservations[clientId]
          : [];
        const reservationIds = reservs.length > 0 ? reservs.map((r) => r?.id || r) : undefined;
        const overridePayload = buildOverridesPayload(reservs);

        const payload = {
          client_id: clientId,
          period_year: formData.period_year,
          period_month: formData.period_month,
          reservation_ids: reservationIds,
        };

        if (Object.keys(overridePayload).length > 0) {
          payload.overrides = overridePayload;
        }

        result = await generateInvoice(companyId, payload);

        // Ouvrir le PDF dans un nouvel onglet
        if (result.pdf_url) {
          window.open(result.pdf_url, '_blank');
        }

        onInvoiceGenerated(result);
      } else {
        // Facturation tierce (consolidée)
        // NOUVEAU: Préparer le mapping des réservations par client
        const clientReservations = {};
        formData.client_ids.forEach((clientId) => {
          const reservs = selectedReservations?.[clientId];
          if (reservs && Array.isArray(reservs) && reservs.length > 0) {
            clientReservations[clientId] = reservs.map((r) => r?.id || r);
          }
        });

        const overridePayload = buildOverridesPayload(consolidatedSelection);

        const payload = {
          client_ids: formData.client_ids.map((id) => parseInt(id)),
          bill_to_client_id: parseInt(formData.bill_to_client_id),
          period_year: formData.period_year,
          period_month: formData.period_month,
          client_reservations:
            Object.keys(clientReservations).length > 0 ? clientReservations : undefined,
        };

        if (Object.keys(overridePayload).length > 0) {
          payload.overrides = overridePayload;
        }

        result = await invoiceService.generateConsolidatedInvoice(companyId, payload);

        if (result.invoices && result.invoices.length > 0) {
          setSuccessMessage(
            `${result.success_count} facture(s) générée(s) avec succès${
              result.error_count > 0 ? `, ${result.error_count} erreur(s)` : ''
            }`
          );

          // Ouvrir les PDFs dans de nouveaux onglets
          result.invoices.forEach((inv) => {
            if (inv.pdf_url) {
              window.open(inv.pdf_url, '_blank');
            }
          });

          // Notifier le parent pour chaque facture
          result.invoices.forEach((inv) => onInvoiceGenerated(inv));
        }

        if (result.errors && result.errors.length > 0) {
          const errorMessages = result.errors
            .map((e) => `Client ${e.client_id}: ${e.error}`)
            .join('\n');
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
      setError(
        err.response?.data?.error || err.message || 'Erreur lors de la génération de la facture'
      );
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
    <div className="modal-overlay">
      <div className="modal-content modal-xl">
        <div className="modal-header">
          <h2 className="modal-title">Nouvelle facture</h2>
          <button className="modal-close" onClick={handleClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && <div className="alert alert-error mb-md">{error}</div>}

          {successMessage && <div className={styles.success}>{successMessage}</div>}

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
                <label htmlFor="clientSearch" className={styles.label}>
                  Recherche client
                </label>
                <input
                  id="clientSearch"
                  type="search"
                  className={styles.searchInput}
                  placeholder="Nom, prénom ou email"
                  value={clientSearch}
                  onChange={(e) => setClientSearch(e.target.value)}
                  disabled={clientsLoading}
                />
                <small className={styles.hint}>
                  Affiche uniquement les clients avec trajets non facturés.
                </small>
              </div>

              {clientsError && <div className="alert alert-error mb-sm">{clientsError}</div>}

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
                  disabled={loading || clientsLoading}
                >
                  <option value="">Sélectionner un client</option>
                  {allClients.map((client) => (
                    <option key={client.id} value={client.id}>
                      {`${formatClientLabel(client)}${
                        directClient && client.id === directClient.id && clientSearch.trim()
                          ? ' (sélectionné)'
                          : ''
                      }`}
                    </option>
                  ))}
                </select>
                {clientsLoading && <small className={styles.hint}>Chargement des clients…</small>}
                {!clientsLoading && allClients.length === 0 && (
                  <small className={styles.hint}>
                    Aucun client avec transports à facturer pour le moment.
                  </small>
                )}
              </div>

              {/* Sélection des transports pour facturation directe */}
              {formData.client_id && (
                <div className={styles.formGroup}>
                  <div className={styles.sectionHeader}>
                    <label className={styles.label}>Transports à facturer</label>
                    <button
                      type="button"
                      className={styles.toggleBtn}
                      onClick={() => setShowReservationSelection(!showReservationSelection)}
                    >
                      {showReservationSelection ? '▼ Masquer' : '▶ Sélectionner'}
                    </button>
                  </div>

                  {showReservationSelection && (
                    <>
                      <ReservationSelector
                        companyId={companyId}
                        clientId={parseInt(formData.client_id)}
                        clientName={directClient?.full_name || ''}
                        period={{ year: formData.period_year, month: formData.period_month }}
                        billToType="patient"
                        vatConfig={vatConfig}
                        overrides={overrides}
                        preselectedIds={
                          preselectedReservations[parseInt(formData.client_id, 10)] || []
                        }
                        onOverrideChange={handleOverrideChange}
                        onSelectionChange={(reservations) =>
                          handleReservationSelectionChange(
                            parseInt(formData.client_id),
                            reservations
                          )
                        }
                      />
                      {directSelection.length > 0 && (
                        <div className={styles.summaryCard}>
                          <div className={styles.summaryCardRow}>
                            <span>Montant HT</span>
                            <strong>{formatCurrency(directTotals.base)}</strong>
                          </div>
                          {vatConfig.applicable && directTotals.vat > 0 && (
                            <div className={styles.summaryCardRow}>
                              <span>TVA totale</span>
                              <strong>{formatCurrency(directTotals.vat)}</strong>
                            </div>
                          )}
                          <div className={`${styles.summaryCardRow} ${styles.summaryCardTotal}`}>
                            <span>Total TTC</span>
                            <strong>{formatCurrency(directTotals.total)}</strong>
                          </div>
                        </div>
                      )}
                    </>
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
                <label className={styles.label}>Sélection des patients</label>

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
                      {`${client.first_name || ''} ${client.last_name || ''}`.trim() ||
                        client.username}
                    </label>
                  ))}
                </div>
              </div>

              {/* Sélection des transports pour chaque patient sélectionné */}
              {formData.client_ids.length > 0 && formData.period_year && formData.period_month && (
                <div className={styles.formGroup}>
                  <label className={styles.label}>Transports à facturer</label>

                  <div className={styles.patientsWithReservations}>
                    {formData.client_ids.map((clientId) => {
                      const client = clients.find((c) => c.id === clientId);
                      if (!client) return null;

                      const reservationsCount = selectedReservations?.[clientId]?.length || 0;

                      return (
                        <div key={clientId} className={styles.patientSection}>
                          <div className={styles.patientSectionHeader}>
                            <h4 className={styles.patientName}>
                              {`${client.first_name || ''} ${client.last_name || ''}`.trim() ||
                                client.username}
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
                              clientName={
                                client.full_name || `${client.first_name} ${client.last_name}`
                              }
                              period={{ year: formData.period_year, month: formData.period_month }}
                              billToType="clinic"
                              vatConfig={vatConfig}
                              overrides={overrides}
                              preselectedIds={preselectedReservations[clientId] || []}
                              onOverrideChange={handleOverrideChange}
                              onSelectionChange={(reservations) =>
                                handleReservationSelectionChange(clientId, reservations)
                              }
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {consolidatedSelection.length > 0 && (
                <div className={styles.summaryCard}>
                  <div className={styles.summaryCardRow}>
                    <span>Montant HT global</span>
                    <strong>{formatCurrency(consolidatedTotals.base)}</strong>
                  </div>
                  {vatConfig.applicable && consolidatedTotals.vat > 0 && (
                    <div className={styles.summaryCardRow}>
                      <span>TVA totale</span>
                      <strong>{formatCurrency(consolidatedTotals.vat)}</strong>
                    </div>
                  )}
                  <div className={`${styles.summaryCardRow} ${styles.summaryCardTotal}`}>
                    <span>Total TTC</span>
                    <strong>{formatCurrency(consolidatedTotals.total)}</strong>
                  </div>
                </div>
              )}

              <small className={styles.hint}>
                {formData.client_ids.length} patient(s) sélectionné(s) •{' '}
                {Object.values(selectedReservations || {}).reduce(
                  (sum, res) => sum + (res?.length || 0),
                  0
                )}{' '}
                transport(s) au total
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

          <div className="modal-footer">
            <button
              type="button"
              onClick={handleClose}
              className="btn btn-secondary"
              disabled={loading}
            >
              Annuler
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={
                loading ||
                (billingType === 'direct' && !formData.client_id) ||
                (billingType === 'third_party' &&
                  (formData.client_ids.length === 0 || !formData.bill_to_client_id))
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
