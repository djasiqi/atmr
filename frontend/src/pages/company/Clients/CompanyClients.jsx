import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
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
} from 'react-icons/fi';
import {
  fetchCompanyClients,
  createClient,
  updateClient,
  deleteClient,
  fetchClientDetails,
  createClientStay,
  linkClientBillingParty,
} from '../../../services/companyService';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import ClientsTable from './components/ClientsTable';
import EditClientModal from './components/EditClientModal';
import NewClientModal from './components/NewClientModal';
import DeleteConfirmModal from './components/DeleteConfirmModal';
import ClientReadView from './components/ClientReadView';
import ClientEditForm from './components/ClientEditForm';
import styles from './CompanyClients.module.css';
import useUrlSearchSync from '../../../hooks/useUrlSearchSync';
import {
  normalizeText,
  buildSearchHaystack,
  getClientDisplayName,
} from '../../../utils/clientSearchUtils';

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
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [editingClient, setEditingClient] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showNewClientModal, setShowNewClientModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [clientToDelete, setClientToDelete] = useState(null);

  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [sortBy, setSortBy] = useState('name');
  const [sortOrder, setSortOrder] = useState('asc');

  const [selectedClientId, setSelectedClientId] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [clientDetails, setClientDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [pendingOpenClientId, setPendingOpenClientId] = useState(null);

  const [searchParams, setSearchParams] = useSearchParams();
  const didInitOpenClientRef = useRef(false);
  const searchInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();

  const clientCache = useMemo(() => new Map(), []);

  const loadClients = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchCompanyClients();
      const clientsArray = Array.isArray(data) ? data : [];
      setClients(clientsArray);
    } catch (err) {
      console.error('Erreur lors du chargement des clients:', err);
      setError('Impossible de charger les clients');
      setClients([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadClients();
  }, [loadClients]);

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

  const clientsWithHaystack = useMemo(
    () =>
      clients.map((c) => ({
        client: c,
        haystackNorm: normalizeText(buildSearchHaystack(c)),
      })),
    [clients]
  );

  const filteredAndSortedClients = useMemo(() => {
    const qNorm = searchTerm ? normalizeText(searchTerm) : '';
    let filtered = clientsWithHaystack.filter(({ client, haystackNorm }) => {
      const matchesSearch = !qNorm || haystackNorm.includes(qNorm);
      const matchesType =
        filterType === 'all'
          ? true
          : filterType === 'institution'
            ? client.is_institution
            : !client.is_institution;
      return matchesSearch && matchesType;
    });

    const list = filtered.map(({ client }) => client);
    list.sort((a, b) => {
      let compareA, compareB;

      switch (sortBy) {
        case 'name': {
          const getNameKey = (c) => {
            if (c.is_institution) return getClientDisplayName(c).toLowerCase();
            const last = (c.last_name || '').toLowerCase();
            const first = (c.first_name || '').toLowerCase();
            return last ? `${last} ${first}` : getClientDisplayName(c).toLowerCase();
          };
          compareA = getNameKey(a);
          compareB = getNameKey(b);
          break;
        }
        case 'created':
          compareA = new Date(a.created_at || 0);
          compareB = new Date(b.created_at || 0);
          break;
        default:
          return 0;
      }

      if (compareA < compareB) return sortOrder === 'asc' ? -1 : 1;
      if (compareA > compareB) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });

    return list;
  }, [clientsWithHaystack, searchTerm, filterType, sortBy, sortOrder]);

  const totalPages = Math.ceil(filteredAndSortedClients.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedClients = filteredAndSortedClients.slice(startIndex, endIndex);

  React.useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, filterType]);

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

    setLoadingDetails(true);
    try {
      const details = await fetchClientDetails(clientId);
      clientCache.set(clientId, details);
      setClientDetails(details);
    } catch (err) {
      console.error('Erreur lors du chargement des details:', err);
      setError('Erreur lors du chargement des details du client');
    } finally {
      setLoadingDetails(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSelectClient = useCallback((client) => {
    const clientId = client.id;
    setSelectedClientId(clientId);
    setIsDrawerOpen(true);
    setIsEditMode(false);
    setHasUnsavedChanges(false);

    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set('selected', clientId.toString());
    setSearchParams(newSearchParams);

    loadClientDetails(clientId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadClientDetails]);

  const handleCloseDrawer = useCallback(() => {
    if (hasUnsavedChanges) {
      const confirmed = window.confirm(
        'Modifications non sauvegardees. Voulez-vous vraiment fermer ?'
      );
      if (!confirmed) return;
    }

    setIsDrawerOpen(false);
    setSelectedClientId(null);
    setPendingOpenClientId(null);
    setIsEditMode(false);
    setHasUnsavedChanges(false);
    setClientDetails(null);

    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.delete('selected');
    newSearchParams.delete('openClientId');
    newSearchParams.delete('clientId');
    setSearchParams(newSearchParams);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    const selectedId = searchParams.get('selected');
    if (selectedId) {
      const clientId = parseInt(selectedId, 10);
      if (!isNaN(clientId) && clientId !== selectedClientId) {
        const client = clients.find((c) => c.id === clientId);
        if (client) {
          handleSelectClient(client);
        }
      }
    }
  }, [searchParams, clients, selectedClientId, handleSelectClient]);

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
    total: clients.length,
    regular: clients.filter((c) => !c.is_institution).length,
    institutions: clients.filter((c) => c.is_institution).length,
    active: clients.filter((c) => c.is_active).length,
  }), [clients]);

  const panelOpen = isDrawerOpen && clientDetails;

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
      <CompanyHeader />
      <div className={styles.layout}>
        <CompanySidebar />
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
                <button type="button" className={styles.refreshBtn} title="Rafraîchir" onClick={loadClients}>
                  <FiRefreshCw size={14} />
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

            {/* Loading / Error */}
            {loading && <div className={styles.loading}>Chargement des clients...</div>}

            {error && (
              <div className={styles.error}>
                <span>{error}</span>
                <button onClick={loadClients} className={styles.retryBtn}>
                  <FiRefreshCw size={14} />
                  Reessayer
                </button>
              </div>
            )}

            {!loading && !error && (
              <>
                {/* Zone D -- Table */}
                <ClientsTable
                  clients={paginatedClients}
                  onSelect={handleSelectClient}
                  onEdit={handleEditClient}
                  onDelete={handleDeleteClick}
                  selectedClientId={selectedClientId}
                  onRefresh={loadClients}
                />

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
                {isEditMode ? (
                  <ClientEditForm
                    client={clientDetails}
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
                ) : (
                  <ClientReadView
                    client={clientDetails}
                    onEdit={handleEditInDrawer}
                    onClose={handleCloseDrawer}
                    loading={loadingDetails}
                  />
                )}
              </div>
            </aside>
          )}
        </div>
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
