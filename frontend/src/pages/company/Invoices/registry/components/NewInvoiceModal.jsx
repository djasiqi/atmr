import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import styles from './NewInvoiceModal.module.css';
import { generateInvoice, invoiceService } from '../../../../../services/invoiceService';
import ReservationSelector from './ReservationSelector';

const NewInvoiceModal = ({ open, onClose, onInvoiceGenerated, companyId, initialDraft = null }) => {
  const [billingType, setBillingType] = useState('direct'); // 'direct', 'third_party' ou 'partner'
  const [formData, setFormData] = useState({
    client_id: '',
    client_ids: [],
    bill_to_client_id: '',
    partnership_id: '',
    period_year: new Date().getFullYear(),
    period_month: new Date().getMonth() + 1,
  });
  const [clients, setClients] = useState([]);
  const [clientCache, setClientCache] = useState({});
  const [clientSearch, setClientSearch] = useState('');
  const [clientsLoading, setClientsLoading] = useState(false);
  const [clientsError, setClientsError] = useState(null);
  // ✅ Référence pour garder le focus sur l'input de recherche
  const clientSearchInputRef = useRef(null);
  const wasInputFocusedRef = useRef(false);
  const [institutions, setInstitutions] = useState([]);
  const [partners, setPartners] = useState([]);
  const [partnersLoading, setPartnersLoading] = useState(false);
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
      const billing = initialDraft.billing_type === 'partner'
        ? 'partner'
        : initialDraft.billing_type === 'third_party'
          ? 'third_party'
          : 'direct';
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
        partnership_id: initialDraft.partnership_id
          ? String(initialDraft.partnership_id)
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

  // Charger la liste des partenaires facturables quand le type est "partner"
  useEffect(() => {
    if (!open || !companyId || billingType !== 'partner') return;

    let isMounted = true;

    const loadPartners = async () => {
      try {
        setPartnersLoading(true);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:loadPartners',message:'Loading partners entry',data:{companyId},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        let response;
        try {
          response = await invoiceService.fetchBillablePartners(companyId);
        } catch (err) {
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:loadPartners',message:'Error fetching partners',data:{errorMessage:err?.message,errorStatus:err?.response?.status,errorData:err?.response?.data,fullError:JSON.stringify(err)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
          // #endregion
          throw err;
        }
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:loadPartners',message:'Response received',data:{hasResponse:!!response,responseType:typeof response,responseKeys:response ? Object.keys(response) : [],responseData:response?.data,responseDataData:response?.data?.data,responseDataType:typeof response?.data,partnersCount:response?.data?.data?.length||0,partners:response?.data?.data},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        if (!isMounted) return;
        // La réponse est {data: [...]}, donc on accède à response.data.data
        const partnersList = response?.data?.data || response?.data || [];
        setPartners(partnersList);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:loadPartners',message:'Partners set in state',data:{partnersCount:partnersList.length,partnersList},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
      } catch (err) {
        console.error('Erreur lors du chargement des partenaires:', err);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:loadPartners',message:'Error loading partners',data:{errorMessage:err?.message,errorStatus:err?.response?.status,errorData:err?.response?.data},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
        // #endregion
        if (isMounted) {
          setError('Erreur lors du chargement des partenaires');
        }
      } finally {
        if (isMounted) {
          setPartnersLoading(false);
        }
      }
    };

    loadPartners();

    return () => {
      isMounted = false;
    };
  }, [companyId, open, billingType]);

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

        console.log('🔍 [NewInvoiceModal] fetchEligibleClients appelé avec:', {
          companyId,
          search: query || undefined,
          limit: 120,
          year: formData.period_year,
          month: formData.period_month,
        });

        const response = await invoiceService.fetchEligibleClients(companyId, {
          search: query || undefined,
          limit: 120,
          year: formData.period_year,
          month: formData.period_month,
        });

        console.log('🔍 [NewInvoiceModal] Réponse reçue:', {
          response,
          responseData: response?.data,
          hasClients: !!(response?.data?.clients),
          clientsType: typeof response?.data?.clients,
          clientsLength: Array.isArray(response?.data?.clients) ? response.data.clients.length : 'N/A',
          clients: response?.data?.clients,
        });

        // Le service retourne response.data, donc response est déjà {clients: [...], total: ...}
        // Mais axios peut avoir une structure imbriquée, donc on vérifie les deux
        const list = Array.isArray(response?.clients) 
          ? response.clients 
          : Array.isArray(response?.data?.clients) 
            ? response.data.clients 
            : [];

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
          // ✅ Refocuser l'input après la mise à jour de la liste pour permettre la saisie continue
          // Utiliser requestAnimationFrame pour s'assurer que le DOM est mis à jour
          requestAnimationFrame(() => {
            if (clientSearchInputRef.current && wasInputFocusedRef.current) {
              clientSearchInputRef.current.focus();
              // ✅ Restaurer la position du curseur à la fin du texte
              const length = clientSearchInputRef.current.value.length;
              clientSearchInputRef.current.setSelectionRange(length, length);
            }
          });
        }
      }
    };

    const timer = setTimeout(fetchClients, 250);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [companyId, open, clientSearch, formData.period_year, formData.period_month]);

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

    if (billingType === 'partner') {
      if (!formData.partnership_id) {
        setError('Veuillez sélectionner un partenaire');
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

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:handleSubmit',message:'Appel generateInvoice',data:{companyId,payload:{client_id:payload.client_id,period_year:payload.period_year,period_month:payload.period_month,has_reservation_ids:!!payload.reservation_ids}},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
        // #endregion

        result = await generateInvoice(companyId, payload);

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:handleSubmit',message:'Réponse generateInvoice reçue',data:{has_result:!!result,result_type:result ? typeof result : 'null',result_keys:result ? Object.keys(result) : [],has_pdf_url:!!result?.pdf_url,has_data:!!result?.data,has_data_pdf_url:!!result?.data?.pdf_url,has_id:!!result?.id,result_stringified:JSON.stringify(result).substring(0,500)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
        // #endregion

        // Ouvrir le PDF dans un nouvel onglet
        if (result?.pdf_url) {
          window.open(result.pdf_url, '_blank');
        } else if (result?.data?.pdf_url) {
          // Si la structure est {data: {pdf_url: ...}}
          window.open(result.data.pdf_url, '_blank');
        }

        // Vérifier la structure de la réponse avant de notifier le parent
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:handleSubmit',message:'Avant onInvoiceGenerated',data:{has_result_data:!!result?.data,has_result:!!result,will_call_with_data:!!result?.data,will_call_with_result:!result?.data && !!result,result_id:result?.id || result?.data?.id,result_period_year:result?.period_year || result?.data?.period_year,result_period_month:result?.period_month || result?.data?.period_month},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
        // #endregion

        if (result?.data) {
          onInvoiceGenerated(result.data);
        } else if (result) {
        onInvoiceGenerated(result);
        }
        
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'NewInvoiceModal.jsx:handleSubmit',message:'Après onInvoiceGenerated',data:{called:true},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
        // #endregion
      } else if (billingType === 'third_party') {
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

        if (result?.errors && result.errors.length > 0) {
          const errorMessages = result.errors
            .map((e) => `Client ${e.client_id}: ${e.error}`)
            .join('\n');
          setError(`Erreurs:\n${errorMessages}`);
        }
      } else if (billingType === 'partner') {
        // Facturation partenaire
        const payload = {
          partnership_id: parseInt(formData.partnership_id),
          period_year: formData.period_year,
          period_month: formData.period_month,
        };

        result = await invoiceService.generatePartnerInvoice(companyId, payload);

        // Ouvrir le PDF dans un nouvel onglet
        if (result?.data?.pdf_url) {
          window.open(result.data.pdf_url, '_blank');
        }

        // Vérifier que result.data existe avant de notifier le parent
        if (result?.data) {
        onInvoiceGenerated(result.data);
        } else if (result) {
          // Si result.data n'existe pas mais result existe, utiliser result directement
          onInvoiceGenerated(result);
        }
      }

      // Fermer le modal si tout s'est bien passé et pas d'erreurs
      if (!result || !result.errors || result.errors.length === 0) {
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
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  value="partner"
                  checked={billingType === 'partner'}
                  onChange={(e) => setBillingType(e.target.value)}
                  disabled={loading}
                />
                Facturation partenaire
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
                  ref={clientSearchInputRef}
                  id="clientSearch"
                  type="search"
                  className={styles.searchInput}
                  placeholder="Nom, prénom ou email"
                  value={clientSearch}
                  onChange={(e) => setClientSearch(e.target.value)}
                  onFocus={() => {
                    wasInputFocusedRef.current = true;
                  }}
                  onBlur={() => {
                    wasInputFocusedRef.current = false;
                  }}
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

          {/* Facturation partenaire */}
          {billingType === 'partner' && (
            <>
              <div className={styles.formGroup}>
                <label htmlFor="partnership_id" className={styles.label}>
                  Partenaire *
                </label>
                <select
                  id="partnership_id"
                  name="partnership_id"
                  value={formData.partnership_id}
                  onChange={handleInputChange}
                  className={styles.select}
                  required
                  disabled={loading || partnersLoading}
                >
                  <option value="">Sélectionner un partenaire</option>
                  {partners.map((partner) => (
                    <option key={partner.partnership_id} value={partner.partnership_id}>
                      {partner.partner_company_name} ({partner.unbilled_transfers_count} transfert{partner.unbilled_transfers_count > 1 ? 's' : ''} • {formatCurrency(partner.total_amount)} {partner.currency})
                    </option>
                  ))}
                </select>
                {partnersLoading && <small className={styles.hint}>Chargement des partenaires…</small>}
                {!partnersLoading && partners.length === 0 && (
                  <small className={styles.hint}>
                    Aucun partenaire avec transferts facturables pour le moment.
                  </small>
                )}
              </div>
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
