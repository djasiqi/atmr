// src/pages/company/Driver/CompanyDriver.jsx
import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  FiPlus,
  FiCalendar,
  FiSearch,
  FiUsers,
  FiCheckCircle,
  FiTruck,
  FiWifiOff,
  FiMapPin,
  FiInbox,
  FiChevronDown,
  FiRefreshCw,
  FiX,
} from 'react-icons/fi';
import { createPortal } from 'react-dom';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import DriverLiveMap from '../Dashboard/components/DriverLiveMap';
import useDriver from '../../../hooks/useDriver';
import CompanyDriverTable from '../components/CompanyDriverTable';
import AddDriverForm from '../components/AddDriverForm';
import EditDriverForm from '../components/EditDriverForm';
import useAuthToken from '../../../hooks/useAuthToken';
import {
  fetchDriverCompletedTrips,
  createDriver,
  updateDriverDetails,
} from '../../../services/companyService';
import DriverWorkingHoursTable from './DriverWorkingHoursTable';
import { toast } from 'sonner';
import s from './CompanyDriver.module.css';

const AVAILABILITY_OPTIONS = [
  { value: 'all', label: 'Tous' },
  { value: 'available', label: 'Disponibles' },
  { value: 'onTrip', label: 'En course' },
  { value: 'offline', label: 'Hors ligne' },
];

const ChipDrop = ({ value, onChange, options, prefix }) => {
  const [open, setOpen] = useState(false);
  const btnRef = React.useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  const toggle = () => {
    if (!open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      setPos({ top: r.bottom + 4, left: r.left });
    }
    setOpen((v) => !v);
  };

  const current = options.find((o) => o.value === value);

  return (
    <div className={s.chipDrop}>
      <button type="button" ref={btnRef} className={`${s.chipBtn} ${open ? s.chipBtnActive : ''}`} onClick={toggle}>
        <span className={s.chipText}>{prefix ? `${prefix} ${current?.label}` : current?.label}</span>
        <FiChevronDown size={11} className={`${s.chipArrow} ${open ? s.chipArrowOpen : ''}`} />
      </button>
      {open && createPortal(
        <>
          <div className={s.chipBackdrop} onClick={() => setOpen(false)} />
          <div className={s.chipMenu} style={{ position: 'fixed', top: pos.top, left: pos.left }}>
            {options.map((opt) => (
              <div
                key={opt.value}
                className={`${s.chipOption} ${opt.value === value ? s.chipOptionActive : ''}`}
                onClick={() => { onChange(opt.value); setOpen(false); }}
              >
                {opt.label}
              </div>
            ))}
          </div>
        </>,
        document.body,
      )}
    </div>
  );
};

const CompanyDriver = () => {
  const user = useAuthToken();
  const { drivers, loading, error, toggleDriverStatus, deleteDriverById, refreshDrivers } =
    useDriver();

  const [showAddDriverModal, setShowAddDriverModal] = useState(false);
  const [showEditDriverModal, setShowEditDriverModal] = useState(false);
  const [driverToEdit, setDriverToEdit] = useState(null);
  const [_driverToDelete, setDriverToDelete] = useState(null);
  const [confirmDialog, setConfirmDialog] = useState({ open: false, title: '', message: '', onConfirm: null });
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [availabilityFilter, setAvailabilityFilter] = useState('all');
  const [activeTab, setActiveTab] = useState('drivers');
  const [driverHoursData, setDriverHoursData] = useState([]);
  const [mapCollapsed, setMapCollapsed] = useState(true);

  // Stats
  const driverStats = useMemo(() => {
    const OFFLINE_THRESHOLD_MS = 30 * 60 * 1000; // 30 minutes
    const now = Date.now();

    const isRecentGps = (d) => {
      if (!d.last_position_update) return false;
      const elapsed = now - new Date(d.last_position_update).getTime();
      return elapsed < OFFLINE_THRESHOLD_MS;
    };

    const total = drivers.length;
    const onTrip = drivers.filter((d) => d.current_trip_id).length;
    const located = drivers.filter((d) => d.latitude && d.longitude && isRecentGps(d)).length;
    const available = drivers.filter(
      (d) => d.is_active && !d.current_trip_id && isRecentGps(d)
    ).length;
    const offline = total - onTrip - available;
    return { total, onTrip, available, offline, located };
  }, [drivers]);

  // Filtering
  const filteredDrivers = useMemo(() => {
    return (drivers || []).filter((drv) => {
      const text = `${drv.username} ${drv.first_name} ${drv.last_name} ${drv.email || ''}`.toLowerCase();
      const matchesSearch = text.includes(searchTerm.toLowerCase());

      const matchesStatus =
        statusFilter === 'all' ||
        (statusFilter === 'active' && drv.is_active) ||
        (statusFilter === 'inactive' && !drv.is_active);

      const matchesAvailability =
        availabilityFilter === 'all' ||
        (availabilityFilter === 'available' && drv.is_active && !drv.current_trip_id) ||
        (availabilityFilter === 'onTrip' && !!drv.current_trip_id) ||
        (availabilityFilter === 'offline' && !drv.is_active);

      return matchesSearch && matchesStatus && matchesAvailability;
    });
  }, [drivers, searchTerm, statusFilter, availabilityFilter]);

  // Working hours data
  useEffect(() => {
    async function loadStats() {
      const mappedData = [];
      const BATCH_SIZE = 10;
      for (let i = 0; i < drivers.length; i += BATCH_SIZE) {
        const batch = drivers.slice(i, i + BATCH_SIZE);
        const batchPromises = batch.map(async (drv) => {
          let trips = [];
          try {
            trips = await fetchDriverCompletedTrips(drv.id);
          } catch {
            trips = [];
          }
          const count = trips.length;
          const totalMinutes = trips.reduce((sum, trip) => sum + (trip.duration_in_minutes || 0), 0);
          return { driverId: drv.id, driverName: drv.username, count, totalMinutes };
        });
        const batchResults = await Promise.all(batchPromises);
        mappedData.push(...batchResults);
      }
      setDriverHoursData(mappedData);
    }
    if (drivers && drivers.length > 0) {
      loadStats();
    } else {
      setDriverHoursData([]);
    }
  }, [drivers]);

  // Handlers
  const handleAddSubmit = useCallback(async (payload) => {
    try {
      await createDriver(payload);
      refreshDrivers();
      setShowAddDriverModal(false);
      toast.success('Chauffeur ajoute avec succes');
    } catch (err) {
      console.error("Erreur lors de l'ajout du chauffeur:", err);
      let errorMessage = err?.message || err?.error || 'Une erreur est survenue.';
      if (err?.status === 409 || errorMessage.includes('Conflict')) {
        errorMessage = "Ce nom d'utilisateur ou cet email est deja utilise.";
      }
      toast.error(errorMessage);
    }
  }, [refreshDrivers]);

  const handleEditSubmit = useCallback(async (driverId, payload) => {
    try {
      await updateDriverDetails(driverId, payload);
      refreshDrivers();
      setShowEditDriverModal(false);
      toast.success('Chauffeur mis a jour');
    } catch (err) {
      console.error('Erreur de mise a jour:', err);
      toast.error(err.error || 'Veuillez reessayer.');
    }
  }, [refreshDrivers]);

  const handleDeleteRequest = useCallback((driver) => {
    setDriverToDelete(driver);
    setConfirmDialog({
      open: true,
      title: 'Supprimer le chauffeur',
      message: `Êtes-vous sûr de vouloir supprimer définitivement le chauffeur ${driver?.username} ?`,
      onConfirm: async () => {
        await deleteDriverById(driver.id);
        setConfirmDialog((d) => ({ ...d, open: false }));
        setDriverToDelete(null);
        toast.success('Chauffeur supprimé');
      },
    });
  }, [deleteDriverById]);

  const openEditModal = useCallback((driver) => {
    setShowAddDriverModal(false);
    setDriverToEdit(driver);
    setShowEditDriverModal(true);
  }, []);

  const openAddPanel = useCallback(() => {
    setShowEditDriverModal(false);
    setDriverToEdit(null);
    setShowAddDriverModal(true);
  }, []);

  const closeSidePanel = useCallback(() => {
    setShowAddDriverModal(false);
    setShowEditDriverModal(false);
    setDriverToEdit(null);
  }, []);

  const panelOpen = showAddDriverModal || (showEditDriverModal && driverToEdit);

  // Tabs
  const activeCount = drivers.filter((d) => d.is_active).length;
  const inactiveCount = drivers.filter((d) => !d.is_active).length;

  const tabs = useMemo(() => [
    { id: 'all', label: 'Tous', count: drivers.length },
    { id: 'active', label: 'Actifs', count: activeCount },
    { id: 'inactive', label: 'Inactifs', count: inactiveCount },
  ], [drivers.length, activeCount, inactiveCount]);

  const handleTabChange = useCallback((tabId) => {
    setStatusFilter(tabId);
  }, []);

  return (
    <div className={s.pageContainer}>
      <CompanyHeader />
      <div className={s.layout}>
        <CompanySidebar />
        <div className={`${s.contentArea} ${panelOpen ? s.contentAreaWithPanel : ''}`}>
        <main className={s.main}>
          {error && <div className={s.errorBanner}>{error}</div>}

          {/* ===== ZONE A — Page Header ===== */}
          <div className={s.pageHeader}>
            <div className={s.pageHeaderLeft}>
              <h1 className={s.pageTitle}>Chauffeurs</h1>
              <p className={s.pageSubtitle}>
                Gerez votre equipe et suivez leur disponibilite en temps reel
              </p>
            </div>
            <div className={s.pageHeaderActions}>
              <Link
                to={`/dashboard/company/${user?.public_id || user?.id}/driver/planning`}
                className={s.btnSecondary}
              >
                <FiCalendar size={16} />
                Voir le planning
              </Link>
              <button
                type="button"
                className={s.btnPrimary}
                onClick={openAddPanel}
              >
                <FiPlus size={16} />
                Ajouter un chauffeur
              </button>
            </div>
          </div>

          {/* ===== ZONE B — Command Bar ===== */}
          <div className={s.commandBar}>
            <div className={s.searchWrap}>
              <FiSearch size={14} className={s.searchIcon} />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Rechercher..."
                className={s.searchInput}
              />
              {searchTerm && (
                <button type="button" className={s.clearBtn} onClick={() => setSearchTerm('')}>
                  <FiX size={11} />
                </button>
              )}
            </div>

            <div className={s.segmented}>
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`${s.segBtn} ${statusFilter === tab.id ? s.segBtnActive : ''}`}
                  onClick={() => handleTabChange(tab.id)}
                >
                  {tab.label} ({tab.count})
                </button>
              ))}
            </div>

            <ChipDrop
              value={availabilityFilter}
              onChange={setAvailabilityFilter}
              options={AVAILABILITY_OPTIONS}
              prefix="Disponibilité :"
            />

            <div className={s.barSpacer} />

            <div className={s.barMeta}>
              <span className={s.barResultCount}>{filteredDrivers.length} résultat{filteredDrivers.length !== 1 ? 's' : ''}</span>
              <button type="button" className={s.refreshBtn} title="Rafraîchir" onClick={refreshDrivers}>
                <FiRefreshCw size={14} />
              </button>
            </div>
          </div>

          {/* ===== ZONE C — KPIs + Map ===== */}
          <div className={s.opsSection}>
            {/* KPIs */}
            <div className={s.kpiRow}>
              <div className={s.kpiCard}>
                <div className={`${s.kpiIcon} ${s.kpiIconTotal}`}>
                  <FiUsers size={18} />
                </div>
                <div className={s.kpiBody}>
                  <span className={s.kpiValue}>{driverStats.total}</span>
                  <span className={s.kpiLabel}>Total</span>
                </div>
              </div>
              <div className={s.kpiCard}>
                <div className={`${s.kpiIcon} ${s.kpiIconAvailable}`}>
                  <FiCheckCircle size={18} />
                </div>
                <div className={s.kpiBody}>
                  <span className={s.kpiValue}>{driverStats.available}</span>
                  <span className={s.kpiLabel}>Disponibles</span>
                </div>
              </div>
              <div className={s.kpiCard}>
                <div className={`${s.kpiIcon} ${s.kpiIconOnTrip}`}>
                  <FiTruck size={18} />
                </div>
                <div className={s.kpiBody}>
                  <span className={s.kpiValue}>{driverStats.onTrip}</span>
                  <span className={s.kpiLabel}>En course</span>
                </div>
              </div>
              <div className={s.kpiCard}>
                <div className={`${s.kpiIcon} ${s.kpiIconOffline}`}>
                  <FiWifiOff size={18} />
                </div>
                <div className={s.kpiBody}>
                  <span className={s.kpiValue}>{driverStats.offline}</span>
                  <span className={s.kpiLabel}>Hors ligne</span>
                </div>
              </div>
            </div>

            {/* Map */}
            <div className={s.mapCard}>
              <div className={s.mapCardHeader}>
                <div className={s.mapHeaderLeft}>
                  <FiMapPin size={16} className={s.mapHeaderIcon} />
                  <span className={s.mapHeaderTitle}>Localisation en direct</span>
                  <span className={s.mapLocatedBadge}>
                    {driverStats.located}/{driverStats.total} localises
                  </span>
                  {driverStats.located === 0 && driverStats.total > 0 && (
                    <span className={s.mapNoGps}>Aucun GPS recent</span>
                  )}
                </div>
                <div className={s.mapHeaderRight}>
                  {!mapCollapsed && (
                    <>
                      <span className={s.liveDot} />
                      <span className={s.liveText}>En direct</span>
                    </>
                  )}
                  <button
                    type="button"
                    className={s.collapseBtn}
                    onClick={() => setMapCollapsed((v) => !v)}
                  >
                    {mapCollapsed ? 'Afficher' : 'Masquer'}
                  </button>
                </div>
              </div>
              {!mapCollapsed && (
                <div className={s.mapWrap}>
                  <DriverLiveMap drivers={drivers} />
                  {driverStats.located === 0 && driverStats.total > 0 && (
                    <div className={s.mapEmpty}>
                      <FiMapPin size={24} />
                      <p>Aucun chauffeur localise</p>
                      <span>Les positions apparaitront des que les chauffeurs activeront leur GPS</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* ===== ZONE D — Content tabs (list / hours) ===== */}
          <div className={s.contentTabs}>
            <button
              type="button"
              className={`${s.contentTab} ${activeTab === 'drivers' ? s.contentTabActive : ''}`}
              onClick={() => setActiveTab('drivers')}
            >
              Liste des chauffeurs
              <span className={s.contentTabCount}>{filteredDrivers.length}</span>
            </button>
            <button
              type="button"
              className={`${s.contentTab} ${activeTab === 'hours' ? s.contentTabActive : ''}`}
              onClick={() => setActiveTab('hours')}
            >
              Heures effectuees
            </button>
          </div>

          {activeTab === 'drivers' && (
            <>
              {loading ? (
                <div className={s.loadingState}>
                  <div className={s.spinner} />
                  <p>Chargement des chauffeurs...</p>
                </div>
              ) : filteredDrivers.length === 0 ? (
                <div className={s.emptyState}>
                  <FiInbox size={36} className={s.emptyIcon} />
                  <h3 className={s.emptyTitle}>Aucun chauffeur dans cette categorie</h3>
                  <p className={s.emptySubtitle}>
                    Ajustez vos filtres ou ajoutez un nouveau chauffeur.
                  </p>
                  <button
                    type="button"
                    className={s.btnPrimary}
                    onClick={openAddPanel}
                  >
                    <FiPlus size={14} />
                    Ajouter un chauffeur
                  </button>
                </div>
              ) : (
                <CompanyDriverTable
                  drivers={filteredDrivers}
                  onEdit={openEditModal}
                  onToggleStatus={toggleDriverStatus}
                  onDeleteRequest={handleDeleteRequest}
                />
              )}
            </>
          )}

          {activeTab === 'hours' && <DriverWorkingHoursTable driverHoursData={driverHoursData} />}

          {confirmDialog.open && (
            <div className={s.confirmOverlay} onClick={() => setConfirmDialog((d) => ({ ...d, open: false }))}>
              <div className={s.confirmModal} onClick={(e) => e.stopPropagation()}>
                <h3 className={s.confirmTitle}>{confirmDialog.title}</h3>
                <p className={s.confirmMessage}>{confirmDialog.message}</p>
                <div className={s.confirmActions}>
                  <button
                    type="button"
                    className={s.confirmCancel}
                    onClick={() => setConfirmDialog((d) => ({ ...d, open: false }))}
                  >
                    Annuler
                  </button>
                  <button
                    type="button"
                    className={`${s.confirmBtn} ${s.confirmBtnDanger}`}
                    onClick={confirmDialog.onConfirm}
                  >
                    Confirmer
                  </button>
                </div>
              </div>
            </div>
          )}
        </main>

        {/* Side Panel */}
        {panelOpen && (
          <aside className={s.sidePanel}>
            <div className={s.sidePanelInner}>
              {showAddDriverModal && (
                <AddDriverForm onSubmit={handleAddSubmit} onClose={closeSidePanel} />
              )}
              {showEditDriverModal && driverToEdit && (
                <EditDriverForm
                  key={driverToEdit.id}
                  driver={driverToEdit}
                  onSubmit={handleEditSubmit}
                  onClose={closeSidePanel}
                />
              )}
            </div>
          </aside>
        )}
        </div>
      </div>
    </div>
  );
};

export default CompanyDriver;
