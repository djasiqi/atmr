import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import {
  FiPlus,
  FiSearch,
  FiArrowUp,
  FiArrowDown,
  FiUsers,
  FiUser,
  FiHome,
  FiCheckCircle,
  FiChevronLeft,
  FiChevronRight,
  FiChevronDown,
  FiRefreshCw,
  FiX,
  FiList,
  FiDownload,
} from 'react-icons/fi';
import {
  fetchCompanyClientsPaginated,
  exportCompanyClientsCsv,
  createClient,
  updateClient,
  deleteClient,
  fetchClientDetails,
  createClientStay,
  linkClientBillingParty,
} from '../../../services/companyService';
import { toast } from 'sonner';
import ClientsTable from './components/ClientsTable';
import ClientsTableSkeleton from './components/ClientsTableSkeleton';
import { useLirieCompany } from '../../../hooks/useLirieCompany';
import { lirieKeys, LIRIE_QK_PREFIX } from '../../../queryKeys/lirie';
import EditClientModal from './components/EditClientModal';
import NewClientModal from './components/NewClientModal';
import DeleteConfirmModal from './components/DeleteConfirmModal';
import ClientReadView from './components/ClientReadView';
import ClientEditForm from './components/ClientEditForm';
import styles from './CompanyClients.module.css';
import useUrlSearchSync from '../../../hooks/useUrlSearchSync';

/** Garde la fiche la plus ancienne (id minimal) par `linked_institution_id` (établissements). */
function dedupeInstitutionClientsByLinkedId(list) {
  if (!Array.isArray(list) || list.length === 0) return list;
  const seen = new Set();
  const out = [];
  for (const c of [...list].sort((a, b) => (a.id || 0) - (b.id || 0))) {
    if (c.is_institution && c.linked_institution_id != null) {
      const k = c.linked_institution_id;
      if (seen.has(k)) continue;
      seen.add(k);
    }
    out.push(c);
  }
  return out;
}

function ChipDropdown({ icon, value, options, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const selected = options.find((o) => String(o.value) === String(value));

  return (
    <div className={styles.chipDrop} ref={ref}>
      <button
        type="button"
        className={styles.chipBtn}
        onClick={() => setOpen((p) => !p)}
      >
        {icon}
        <span className={styles.chipText}>{selected?.label || '—'}</span>
        <FiChevronDown size={11} className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`} />
      </button>
      {open && (
        <div className={styles.chipMenu}>
          {options.map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${styles.chipOption} ${String(o.value) === String(value) ? styles.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value); setOpen(false); }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const CompanyClients = () => {
  const { company } = useLirieCompany();
  const canLoadClients = Boolean(company?.id);

  const [detailsError, setDetailsError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [editingClient, setEditingClient] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showNewClientModal, setShowNewClientModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [clientToDelete, setClientToDelete] = useState(null);

  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(25);
  const [sortBy, setSortBy] = useState('name');
  const [sortOrder, setSortOrder] = useState('asc');
  const [isExporting, setIsExporting] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => {
      const next = (searchTerm || '').trim().slice(0, 100);
      if (next.length === 1) {
        setDebouncedSearchTerm('');
        return;
      }
      setDebouncedSearchTerm(next);
    }, 350);
    return () => clearTimeout(t);
  }, [searchTerm]);

  const {
    data: clientsPage,
    isLoading: clientsListInitialLoading,
    isRefetching: clientsListRefetching,
    isError: clientsListError,
    refetch: refetchClients,
  } = useQuery({
    queryKey: canLoadClients
      ? [
          ...lirieKeys.companyClients(company.id),
          currentPage,
          itemsPerPage,
          debouncedSearchTerm,
          sortBy,
          sortOrder,
        ]
      : [LIRIE_QK_PREFIX, 'company-clients', 'disabled'],
    enabled: canLoadClients,
    queryFn: async ({ signal }) => {
      const perPage = Math.min(Math.max(Number(itemsPerPage) || 25, 1), 50);
      return fetchCompanyClientsPaginated({
        page: Math.max(Number(currentPage) || 1, 1),
        perPage,
        search: debouncedSearchTerm,
        sortBy,
        sortOrder,
        signal,
      });
    },
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  });

  const clients = useMemo(() => {
    const rows = clientsPage?.data || clientsPage?.clients || [];
    return Array.isArray(rows) ? rows : [];
  }, [clientsPage]);

  const serverTotal = clientsPage?.pagination?.total ?? clients.length;

  const [selectedClientId, setSelectedClientId] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [clientDetails, setClientDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [pendingOpenClientId, setPendingOpenClientId] = useState(null);

  const [searchParams, setSearchParams] = useSearchParams();
  const selectedClientIdRef = useRef(null);
  /** Après fermeture manuelle : un tour d'effet ignore ?selected= (URL souvent en retard sur l'état), sinon on rouvre la fiche. */
  const skipUrlReopenRef = useRef(false);
  const detailsRequestSeq = useRef(0);
  const didInitOpenClientRef = useRef(false);
  const searchInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();

  const clientCache = useMemo(() => new Map(), []);

  const clientsDeduped = useMemo(
    () => dedupeInstitutionClientsByLinkedId(clients),
    [clients],
  );

  const loadClients = useCallback(async () => {
    setDetailsError(null);
    await refetchClients();
  }, [refetchClients]);

  // Export CSV streamé — indépendant de la liste UI paginée (Lot 5 perf).
  const handleExportCsv = useCallback(async () => {
    setIsExporting(true);
    try {
      await exportCompanyClientsCsv({ search: debouncedSearchTerm });
    } catch (err) {
      console.error('Erreur export CSV clients:', err);
      toast.error("Erreur lors de l'export CSV");
    } finally {
      setIsExporting(false);
    }
  }, [debouncedSearchTerm]);

  useEffect(() => {
    if (!initialized) return;

    if (initialSearch && initialSearch !== searchTerm) {
      setSearchTerm(initialSearch);
    }

    if (shouldFocus) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      requestAnimationFrame(() => {
        searchInputRef.current?.focus();
      });
      consumeFocus();
    }
  }, [initialized, initialSearch, shouldFocus, consumeFocus, searchTerm]);

  useEffect(() => {
    if (didInitOpenClientRef.current) return;
    didInitOpenClientRef.current = true;

    const openClientParam = searchParams.get('openClientId') || searchParams.get('clientId');
    if (openClientParam) {
      const parsedId = parseInt(openClientParam, 10);
      if (!Number.isNaN(parsedId)) {
        setPendingOpenClientId(parsedId);
      }
    }
  }, [searchParams]);

  const filteredAndSortedClients = useMemo(() => {
    // Tri/recherche serveur ; filtre type institution appliqué sur la page courante
    return clientsDeduped.filter((client) => {
      if (filterType === 'all') return true;
      if (filterType === 'institution') return Boolean(client.is_institution);
      return !client.is_institution;
    });
  }, [clientsDeduped, filterType]);

  const totalPages = Math.max(
    1,
    Math.ceil((Number(serverTotal) || filteredAndSortedClients.length) / itemsPerPage)
  );
  const paginatedClients = filteredAndSortedClients;
  // Pagination serveur : indices affichés dérivés de la page/taille courantes (pas d'un slice local).
  const startIndex = Math.max(0, (currentPage - 1) * itemsPerPage);
  const endIndex = startIndex + paginatedClients.length;

  React.useEffect(() => {
    setCurrentPage(1);
  }, [debouncedSearchTerm, filterType, itemsPerPage, sortBy, sortOrder]);

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const loadClientDetails = useCallback(async (clientId, forceRefresh = false) => {
    if (!forceRefresh && clientCache.has(clientId)) {
      setClientDetails(clientCache.get(clientId));
      return;
    }

    const seq = ++detailsRequestSeq.current;
    setLoadingDetails(true);
    try {
      const details = await fetchClientDetails(clientId);
      if (seq !== detailsRequestSeq.current) return;
      clientCache.set(clientId, details);
      setClientDetails(details);
      setDetailsError(null);
    } catch (err) {
      if (seq !== detailsRequestSeq.current) return;
      console.error('Erreur lors du chargement des details:', err);
      setDetailsError('Erreur lors du chargement des details du client');
    } finally {
      if (seq === detailsRequestSeq.current) {
        setLoadingDetails(false);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSelectClient = useCallback((client) => {
    const clientId = client.id;
    skipUrlReopenRef.current = false;
    selectedClientIdRef.current = clientId;
    setSelectedClientId(clientId);
    setIsDrawerOpen(true);
    setIsEditMode(false);
    setHasUnsavedChanges(false);

    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set('selected', String(clientId));
      return next;
    });

    loadClientDetails(clientId);
  }, [loadClientDetails, setSearchParams]);

  const handleCloseDrawer = useCallback(() => {
    if (hasUnsavedChanges) {
      const confirmed = window.confirm(
        'Modifications non sauvegardees. Voulez-vous vraiment fermer ?'
      );
      if (!confirmed) return;
    }

    skipUrlReopenRef.current = true;
    setIsDrawerOpen(false);
    selectedClientIdRef.current = null;
    setSelectedClientId(null);
    setPendingOpenClientId(null);
    setIsEditMode(false);
    setHasUnsavedChanges(false);
    setClientDetails(null);

    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.delete('selected');
      next.delete('openClientId');
      next.delete('clientId');
      return next;
    });
  }, [hasUnsavedChanges, setSearchParams]);

  const handleEditInDrawer = useCallback(() => {
    setIsEditMode(true);
    setHasUnsavedChanges(false);
  }, []);

  const handleCancelEdit = useCallback(() => {
    if (hasUnsavedChanges) {
      const confirmed = window.confirm(
        'Vous avez des modifications non sauvegardees. Voulez-vous vraiment annuler ?'
      );
      if (!confirmed) return;
    }

    setIsEditMode(false);
    setHasUnsavedChanges(false);
    if (selectedClientId) {
      clientCache.delete(selectedClientId);
      loadClientDetails(selectedClientId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadClientDetails]);

  const handleSaveClient = useCallback(async (clientData, clientId = null) => {
    const targetClientId = clientId || editingClient?.id;
    if (!targetClientId) return;

    try {
      await updateClient(targetClientId, clientData);
      await loadClients();

      if (showEditModal) {
        setShowEditModal(false);
        setEditingClient(null);
      }

      if (targetClientId) {
        clientCache.delete(targetClientId);
        if (selectedClientId === targetClientId && isDrawerOpen) {
          await loadClientDetails(targetClientId, true);
        }
      }
    } catch (err) {
      console.error('Erreur lors de la sauvegarde:', err);
      throw err;
    }
  }, [editingClient, showEditModal, clientCache, selectedClientId, isDrawerOpen, loadClients, loadClientDetails]);

  const handleSaveInDrawer = useCallback(async (clientData) => {
    if (!selectedClientId) return;

    try {
      await handleSaveClient(clientData, selectedClientId);
      setIsEditMode(false);
      setHasUnsavedChanges(false);
    } catch (err) {
      console.error('Erreur lors de la sauvegarde:', err);
      throw err;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handleSaveClient]);

  const handleEditClient = (client) => {
    if (selectedClientId === client.id && isDrawerOpen) {
      handleEditInDrawer();
    } else {
      setEditingClient(client);
      setShowEditModal(true);
    }
  };

  useEffect(() => {
    selectedClientIdRef.current = selectedClientId;
  }, [selectedClientId]);

  /* Sync *depuis* l'URL (deep link, back/forward) uniquement : ne pas mettre
   * `selectedClientId` dans les deps, sinon on ré-applique l'ancien ?selected=
   * avant la mise à jour d'URL après un clic, ce qui remet l'ex-client. */
  useEffect(() => {
    if (skipUrlReopenRef.current) {
      skipUrlReopenRef.current = false;
      const selectedId = searchParams.get('selected');
      if (selectedId) {
        setSearchParams((prev) => {
          const next = new URLSearchParams(prev);
          next.delete('selected');
          return next;
        });
      }
      return;
    }

    const selectedId = searchParams.get('selected');
    if (!selectedId) return;
    const clientId = parseInt(selectedId, 10);
    if (Number.isNaN(clientId)) return;
    if (clientId === selectedClientIdRef.current) return;
    const client = clients.find((c) => c.id === clientId);
    if (client) {
      handleSelectClient(client);
    }
  }, [searchParams, clients, handleSelectClient, setSearchParams]);

  useEffect(() => {
    if (!pendingOpenClientId || clients.length === 0) return;
    if (selectedClientId === pendingOpenClientId && isDrawerOpen) {
      setPendingOpenClientId(null);
      return;
    }
    const client = clients.find((c) => c.id === pendingOpenClientId);
    if (client) {
      handleSelectClient(client);
      setPendingOpenClientId(null);
    }
  }, [pendingOpenClientId, clients, selectedClientId, isDrawerOpen, handleSelectClient]);

  const handleDeleteClick = (client) => {
    setClientToDelete(client);
    setShowDeleteModal(true);
  };

  const handleCloseModal = () => {
    setShowEditModal(false);
    setEditingClient(null);
  };

  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
    setClientToDelete(null);
  };

  const handleConfirmDelete = async (hardDelete = false) => {
    if (!clientToDelete) return;

    try {
      await deleteClient(clientToDelete.id, hardDelete);
      await loadClients();
      handleCloseDeleteModal();
    } catch (err) {
      console.error('Erreur lors de la suppression:', err);

      let errorMessage = err.error || err.message || 'Erreur lors de la suppression';

      if (err.reason) {
        errorMessage += '\n\n' + err.reason;
      }

      if (err.suggestion) {
        errorMessage += '\n\n' + err.suggestion;
      }

      alert(errorMessage);
    }
  };

  const handleCreateClient = async (clientData, { existingClient } = {}) => {
    let createdClient = existingClient || null;
    try {
      const { hospitalization, billing_party_link, ...clientPayload } = clientData || {};
      const newClient = createdClient || (await createClient(clientPayload));
      createdClient = newClient;

      const createdClientId = newClient?.id || newClient?.data?.id || newClient?.client?.id;
      if (!createdClientId) {
        throw new Error('Client cree mais identifiant introuvable.');
      }

      if (hospitalization) {
        await createClientStay(createdClientId, {
          company_id: parseInt(hospitalization.company_id, 10),
          start_date: hospitalization.start_date,
          end_date: hospitalization.end_date || null,
          notes: hospitalization.notes || null,
        });
      }

      if (billing_party_link) {
        await linkClientBillingParty(createdClientId, {
          billing_party_id: billing_party_link.billing_party_id,
          role: billing_party_link.role || null,
          is_default: !!billing_party_link.is_default,
          contact_name: billing_party_link.contact_name || null,
          contact_email: billing_party_link.contact_email || null,
          contact_phone: billing_party_link.contact_phone || null,
        });
      }

      await loadClients();
      return newClient;
    } catch (err) {
      console.error('Erreur lors de la creation du client:', err);
      const errorMessage = err?.response?.data?.error || err.message || 'Erreur creation client';
      if (createdClient || err?.createdClient) {
        const wrappedError = new Error(errorMessage);
        wrappedError.createdClient = createdClient || err?.createdClient;
        throw wrappedError;
      }
      throw err;
    }
  };

  const stats = useMemo(() => ({
    total: clientsDeduped.length,
    regular: clientsDeduped.filter((c) => !c.is_institution).length,
    institutions: clientsDeduped.filter((c) => c.is_institution).length,
    active: clientsDeduped.filter((c) => c.is_active).length,
  }), [clientsDeduped]);

  const clientDetailsForPanel = useMemo(() => {
    if (clientDetails == null || selectedClientId == null) return null;
    if (clientDetails.id !== selectedClientId) return null;
    return clientDetails;
  }, [clientDetails, selectedClientId]);

  const panelOpen = isDrawerOpen
    && selectedClientId
    && (clientDetailsForPanel != null || loadingDetails);

  const listError = clientsListError ? 'Impossible de charger les clients' : null;
  const showListSkeleton = !canLoadClients || clientsListInitialLoading;
  const showListRefresh = clientsListRefetching && !showListSkeleton;
  const listOrDetailsError = listError || detailsError;

  useEffect(() => {
    if (!isDrawerOpen) return;

    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        handleCloseDrawer();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isDrawerOpen, handleCloseDrawer]);

  return (
    <>
      <div className={`${styles.contentArea} ${panelOpen ? styles.contentAreaWithPanel : ''}`}>
          <main className={styles.main}>
            {/* Zone A -- Header */}
            <div className={styles.header}>
              <div className={styles.headerLeft}>
                <h1 className={styles.title}>Clients</h1>
                <p className={styles.subtitle}>Gerez vos clients et institutions</p>
              </div>
              <button
                onClick={() => setShowNewClientModal(true)}
                className={styles.addBtn}
              >
                <FiPlus size={16} />
                Ajouter un client
              </button>
            </div>

            {/* Zone B -- CommandBar */}
            <div className={styles.commandBar}>
              <div className={styles.searchWrap}>
                <FiSearch className={styles.searchIcon} size={14} />
                <input
                  type="text"
                  placeholder="Rechercher..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className={styles.searchInput}
                  ref={searchInputRef}
                />
                {searchTerm && (
                  <button className={styles.clearBtn} onClick={() => setSearchTerm('')} type="button">
                    <FiX size={11} />
                  </button>
                )}
              </div>

              <ChipDropdown
                icon={sortOrder === 'asc' ? <FiArrowUp size={12} /> : <FiArrowDown size={12} />}
                value={sortBy}
                options={[
                  { value: 'name', label: 'Nom' },
                  { value: 'created', label: 'Date de création' },
                ]}
                onChange={setSortBy}
              />

              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className={styles.sortOrderBtn}
                title={sortOrder === 'asc' ? 'Ordre croissant' : 'Ordre décroissant'}
              >
                {sortOrder === 'asc' ? <FiArrowUp size={14} /> : <FiArrowDown size={14} />}
              </button>

              <ChipDropdown
                icon={<FiList size={12} />}
                value={itemsPerPage}
                options={[
                  { value: 10, label: '10 / page' },
                  { value: 25, label: '25 / page' },
                  { value: 50, label: '50 / page' },
                ]}
                onChange={(v) => { setItemsPerPage(Number(v)); setCurrentPage(1); }}
              />

              <div className={styles.segmented}>
                <button
                  type="button"
                  className={`${styles.segBtn} ${filterType === 'all' ? styles.segBtnActive : ''}`}
                  onClick={() => setFilterType('all')}
                >
                  Tous
                  <span className={styles.segCount}>{stats.total}</span>
                </button>
                <button
                  type="button"
                  className={`${styles.segBtn} ${filterType === 'regular' ? styles.segBtnActive : ''}`}
                  onClick={() => setFilterType('regular')}
                >
                  Clients
                  <span className={styles.segCount}>{stats.regular}</span>
                </button>
                <button
                  type="button"
                  className={`${styles.segBtn} ${filterType === 'institution' ? styles.segBtnActive : ''}`}
                  onClick={() => setFilterType('institution')}
                >
                  Institutions
                  <span className={styles.segCount}>{stats.institutions}</span>
                </button>
              </div>

              <div className={styles.barSpacer} />

              <div className={styles.barMeta}>
                <span className={styles.barResultCount}>
                  {filteredAndSortedClients.length} résultat{filteredAndSortedClients.length !== 1 ? 's' : ''}
                </span>
                <button
                  type="button"
                  className={styles.refreshBtn}
                  title="Exporter en CSV (tous les résultats filtrés, pas uniquement la page)"
                  onClick={handleExportCsv}
                  aria-busy={isExporting}
                  disabled={isExporting}
                >
                  <FiDownload size={14} />
                </button>
                <button
                  type="button"
                  className={styles.refreshBtn}
                  title="Rafraîchir"
                  onClick={loadClients}
                  aria-busy={clientsListRefetching}
                  disabled={showListSkeleton}
                >
                  <FiRefreshCw size={14} className={clientsListRefetching ? styles.refreshSpin : ''} />
                </button>
              </div>
            </div>

            {/* Zone C -- KPIs inline */}
            <div className={styles.kpiBar}>
              <div className={styles.kpiItem}>
                <FiUsers size={14} className={styles.kpiIcon} />
                <span className={styles.kpiLabel}>Total</span>
                <span className={styles.kpiValue}>{stats.total}</span>
              </div>
              <div className={styles.kpiSep} />
              <div className={styles.kpiItem}>
                <FiUser size={14} className={styles.kpiIcon} />
                <span className={styles.kpiLabel}>Clients</span>
                <span className={styles.kpiValue}>{stats.regular}</span>
              </div>
              <div className={styles.kpiSep} />
              <div className={styles.kpiItem}>
                <FiHome size={14} className={styles.kpiIcon} />
                <span className={styles.kpiLabel}>Institutions</span>
                <span className={styles.kpiValue}>{stats.institutions}</span>
              </div>
              <div className={styles.kpiSep} />
              <div className={styles.kpiItem}>
                <FiCheckCircle size={14} className={styles.kpiIcon} />
                <span className={styles.kpiLabel}>Actifs</span>
                <span className={styles.kpiValue}>{stats.active}</span>
              </div>
            </div>

            {/* Erreur liste (bloque le tableau) ou fiche (message seul) */}
            {listOrDetailsError && (
              <div className={styles.error}>
                <span>{listOrDetailsError}</span>
                {listError && (
                  <button type="button" onClick={loadClients} className={styles.retryBtn}>
                    <FiRefreshCw size={14} />
                    Reessayer
                  </button>
                )}
              </div>
            )}

            {showListSkeleton && <ClientsTableSkeleton rowCount={8} />}

            {!showListSkeleton && !listError && (
              <>
                <div
                  className={showListRefresh ? styles.clientsListRefreshing : styles.clientsListBlock}
                  aria-busy={showListRefresh}
                >
                  {showListRefresh && (
                    <div
                      className={styles.listRefreshBar}
                      role="status"
                      aria-label="Mise à jour des clients"
                    />
                  )}
                  {/* Zone D -- Table */}
                  <ClientsTable
                    clients={paginatedClients}
                    onSelect={handleSelectClient}
                    onEdit={handleEditClient}
                    onDelete={handleDeleteClick}
                    selectedClientId={selectedClientId}
                    onRefresh={loadClients}
                  />
                </div>

                {/* Pagination compacte */}
                {totalPages > 1 && (
                  <div className={styles.paginationBar}>
                    <span className={styles.paginationInfo}>
                      Affichage {startIndex + 1}&ndash;{Math.min(endIndex, filteredAndSortedClients.length)} sur{' '}
                      {filteredAndSortedClients.length} resultats
                    </span>
                    <div className={styles.paginationControls}>
                      <button
                        onClick={() => handlePageChange(currentPage - 1)}
                        disabled={currentPage === 1}
                        className={styles.paginationBtn}
                        aria-label="Page precedente"
                      >
                        <FiChevronLeft size={16} />
                      </button>
                      <span className={styles.pageInfo}>
                        Page {currentPage} sur {totalPages}
                      </span>
                      <button
                        onClick={() => handlePageChange(currentPage + 1)}
                        disabled={currentPage === totalPages}
                        className={styles.paginationBtn}
                        aria-label="Page suivante"
                      >
                        <FiChevronRight size={16} />
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </main>

          {/* Side Panel (inline, same pattern as CompanyDriver) */}
          {panelOpen && (
            <aside className={styles.sidePanel}>
              <div className={styles.sidePanelInner}>
                {loadingDetails && !clientDetailsForPanel ? (
                  <div className={styles.sidePanelLoading}>Chargement de la fiche client…</div>
                ) : isEditMode && clientDetailsForPanel ? (
                  <ClientEditForm
                    key={selectedClientId}
                    client={clientDetailsForPanel}
                    onSave={handleSaveInDrawer}
                    onCancel={handleCancelEdit}
                    onClose={handleCloseDrawer}
                    loading={loadingDetails}
                    hasUnsavedChanges={hasUnsavedChanges}
                    onUnsavedChangesChange={setHasUnsavedChanges}
                    onReloadClient={() => {
                      if (selectedClientId) {
                        clientCache.delete(selectedClientId);
                        loadClientDetails(selectedClientId, true);
                      }
                    }}
                  />
                ) : !isEditMode && clientDetailsForPanel ? (
                  <ClientReadView
                    key={selectedClientId}
                    client={clientDetailsForPanel}
                    onEdit={handleEditInDrawer}
                    onClose={handleCloseDrawer}
                    loading={loadingDetails}
                  />
                ) : null}
              </div>
            </aside>
          )}
      </div>

      {/* Modals (rendered outside layout to avoid z-index issues) */}
      {showEditModal && editingClient && (
        <EditClientModal
          client={editingClient}
          onClose={handleCloseModal}
          onSave={handleSaveClient}
        />
      )}

      {showNewClientModal && (
        <NewClientModal
          onClose={() => setShowNewClientModal(false)}
          onSave={handleCreateClient}
        />
      )}

      {showDeleteModal && clientToDelete && (
        <DeleteConfirmModal
          client={clientToDelete}
          onClose={handleCloseDeleteModal}
          onConfirm={handleConfirmDelete}
        />
      )}
    </>
  );
};

export default CompanyClients;
