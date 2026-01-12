// src/pages/company/Driver/CompanyDriver.jsx
import React, { useEffect, useState } from 'react';
import styles from '../Dashboard/CompanyDashboard.module.css';
import driverStyles from './CompanyDriver.module.css';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import DriverLiveMap from '../Dashboard/components/DriverLiveMap';
import useDriver from '../../../hooks/useDriver';
import CompanyDriverTable from '../components/CompanyDriverTable';
import AddDriverForm from '../components/AddDriverForm';
import EditDriverForm from '../components/EditDriverForm';
import ConfirmationModal from '../../../components/common/ConfirmationModal';
import { Link } from 'react-router-dom';
import useAuthToken from '../../../hooks/useAuthToken';
import {
  fetchDriverCompletedTrips,
  createDriver,
  updateDriverDetails,
} from '../../../services/companyService';
import DriverWorkingHoursTable from './DriverWorkingHoursTable';

const CompanyDriver = () => {
  const user = useAuthToken();
  const { drivers, loading, error, toggleDriverStatus, deleteDriverById, refreshDrivers } =
    useDriver();

  const [showAddDriverModal, setShowAddDriverModal] = useState(false);
  const [showEditDriverModal, setShowEditDriverModal] = useState(false);
  const [driverToEdit, setDriverToEdit] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [driverToDelete, setDriverToDelete] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [activeTab, setActiveTab] = useState('drivers'); // 'drivers' ou 'hours'
  const [driverHoursData, setDriverHoursData] = useState([]);

  const filteredDrivers = (drivers || []).filter((drv) => {
    const matchesSearch = `${drv.username} ${drv.first_name} ${drv.last_name}`
      .toLowerCase()
      .includes(searchTerm.toLowerCase());
    const matchesStatus =
      statusFilter === 'all' ||
      (statusFilter === 'active' && drv.is_active) ||
      (statusFilter === 'inactive' && !drv.is_active);
    return matchesSearch && matchesStatus;
  });

  // 3. Le useEffect utilise maintenant "drivers" pour calculer les stats
  useEffect(() => {
    async function loadStats() {
      const mappedData = [];
      // ✅ Optimisation: Requêtes en parallèle par batch de 10 pour éviter rate limiting
      const BATCH_SIZE = 10;
      for (let i = 0; i < drivers.length; i += BATCH_SIZE) {
        const batch = drivers.slice(i, i + BATCH_SIZE);
        const batchPromises = batch.map(async (drv) => {
          let trips = [];
          try {
            trips = await fetchDriverCompletedTrips(drv.id);
          } catch (e) {
            trips = [];
          }
          const count = trips.length;
          const totalMinutes = trips.reduce((sum, trip) => sum + (trip.duration_in_minutes || 0), 0);
          return {
            driverId: drv.id,
            driverName: drv.username,
            count,
            totalMinutes,
          };
        });
        const batchResults = await Promise.all(batchPromises);
        mappedData.push(...batchResults);
      }
      setDriverHoursData(mappedData);
    }
    if (drivers && drivers.length > 0) {
      // <-- Utilise "drivers"
      loadStats();
    } else {
      setDriverHoursData([]);
    }
  }, [drivers]); // <-- Utilise "drivers"

  const handleAddSubmit = async (payload) => {
    try {
      await createDriver(payload);
      refreshDrivers();
      setShowAddDriverModal(false);
      alert('Chauffeur ajouté avec succès !');
    } catch (err) {
      console.error("Erreur lors de l'ajout du chauffeur:", err);
      
      // Extraire le message d'erreur du backend
      let errorMessage = err?.message || err?.error || 'Une erreur est survenue.';
      
      // Gérer les erreurs 409 (conflit) de manière spécifique
      if (err?.status === 409 || errorMessage.includes('Conflict')) {
        errorMessage = "Ce nom d'utilisateur ou cet email est déjà utilisé. Veuillez en choisir un autre.";
      } else if (err?.details) {
        errorMessage = `${errorMessage}\nDétails: ${err.details}`;
      }
      
      // Afficher l'erreur complète dans la console pour débogage
      console.error("Détails complets de l'erreur:", {
        message: err?.message,
        error: err?.error,
        details: err?.details,
        status: err?.status,
        fullError: err
      });
      
      alert(`Erreur : ${errorMessage}`);
    }
  };

  const handleEditSubmit = async (driverId, payload) => {
    try {
      await updateDriverDetails(driverId, payload);
      refreshDrivers();
      setShowEditDriverModal(false);
      alert('Chauffeur mis à jour avec succès !');
    } catch (err) {
      console.error('Erreur de mise à jour:', err);
      alert(`Erreur : ${err.error || 'Veuillez réessayer.'}`);
    }
  };

  const handleDeleteRequest = (driver) => {
    setDriverToDelete(driver);
    setShowConfirmModal(true);
  };

  const handleConfirmDelete = async () => {
    if (driverToDelete) {
      await deleteDriverById(driverToDelete.id);
      setShowConfirmModal(false);
      setDriverToDelete(null);
    }
  };

  const openEditModal = (driver) => {
    setDriverToEdit(driver);
    setShowEditDriverModal(true);
  };

  // Calcul des statistiques des chauffeurs
  const driverStats = {
    total: drivers.length,
    active: drivers.filter((d) => d.is_active).length,
    onTrip: drivers.filter((d) => d.current_trip_id).length,
    available: drivers.filter((d) => d.is_active && !d.current_trip_id).length,
  };

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {error && <div className={styles.error}>{error}</div>}

          {/* Section Header + Filtres */}
          <section className={driverStyles.headerSection}>
            {/* Header avec titre et bouton */}
            <div className={driverStyles.pageHeader}>
              <div className={driverStyles.headerLeft}>
                <h1>🚗 Gestion des Chauffeurs</h1>
                <p className={driverStyles.subtitle}>
                  Gérez votre équipe de chauffeurs et suivez leur activité en temps réel
                </p>
              </div>
              <button
                className={driverStyles.addButton}
                onClick={() => setShowAddDriverModal(true)}
              >
                <span className={driverStyles.buttonIcon}>➕</span>
                Ajouter Chauffeur
              </button>
            </div>

            {/* Filtres dans le même conteneur */}
            <div className={driverStyles.filterBar}>
              <div className={driverStyles.filterGroup}>
                <label>🔍 Recherche globale</label>
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="ID, nom, email, téléphone, type de chauffeur..."
                  className={driverStyles.searchInput}
                />
              </div>

              <div className={driverStyles.filterButtons}>
                <button
                  className={
                    statusFilter === 'all' ? driverStyles.filterActive : driverStyles.filterBtn
                  }
                  onClick={() => setStatusFilter('all')}
                >
                  Tous ({drivers.length})
                </button>
                <button
                  className={
                    statusFilter === 'active' ? driverStyles.filterActive : driverStyles.filterBtn
                  }
                  onClick={() => setStatusFilter('active')}
                >
                  ✅ Actifs ({drivers.filter((d) => d.is_active).length})
                </button>
                <button
                  className={
                    statusFilter === 'inactive' ? driverStyles.filterActive : driverStyles.filterBtn
                  }
                  onClick={() => setStatusFilter('inactive')}
                >
                  ❌ Inactifs ({drivers.filter((d) => !d.is_active).length})
                </button>
              </div>
            </div>
          </section>

          {/* KPI Cards Section */}
          <div className={driverStyles.statsGrid}>
            <div className={driverStyles.statCard}>
              <span className={driverStyles.statIcon}>👥</span>
              <div className={driverStyles.statContent}>
                <h3 className={driverStyles.statLabel}>Total chauffeurs</h3>
                <p className={driverStyles.statValue}>{driverStats.total}</p>
              </div>
            </div>

            <div className={driverStyles.statCard}>
              <span className={driverStyles.statIcon}>✅</span>
              <div className={driverStyles.statContent}>
                <h3 className={driverStyles.statLabel}>Actifs</h3>
                <p className={driverStyles.statValue}>{driverStats.active}</p>
              </div>
            </div>

            <div className={driverStyles.statCard}>
              <span className={driverStyles.statIcon}>🚗</span>
              <div className={driverStyles.statContent}>
                <h3 className={driverStyles.statLabel}>En course</h3>
                <p className={driverStyles.statValue}>{driverStats.onTrip}</p>
              </div>
            </div>

            <div className={driverStyles.statCard}>
              <span className={driverStyles.statIcon}>⏳</span>
              <div className={driverStyles.statContent}>
                <h3 className={driverStyles.statLabel}>Disponibles</h3>
                <p className={driverStyles.statValue}>{driverStats.available}</p>
              </div>
            </div>
          </div>

          {/* Section Map - Toujours visible */}
          <section className={driverStyles.mapSection}>
            <div className={driverStyles.mapHeader}>
              <h2>📍 Localisation des Chauffeurs</h2>
              <div className={driverStyles.mapControls}>
                <span className={driverStyles.liveIndicator}>
                  <span className={driverStyles.pulse}></span>
                  En direct
                </span>
                <span className={driverStyles.mapInfo}>
                  {drivers.filter((d) => d.latitude && d.longitude).length} / {drivers.length}{' '}
                  localisé(s)
                </span>
              </div>
            </div>
            <div className={driverStyles.mapContainer}>
              <DriverLiveMap drivers={drivers} />
              {drivers.filter((d) => d.latitude && d.longitude).length === 0 &&
                drivers.length > 0 && (
                  <div className={driverStyles.mapOverlay}>
                    <div className={driverStyles.mapOverlayContent}>
                      <span className={driverStyles.mapOverlayIcon}>📍</span>
                      <p>Aucun chauffeur n'a encore partagé sa position GPS</p>
                      <small>
                        Les chauffeurs apparaîtront ici dès qu'ils activeront leur localisation
                      </small>
                    </div>
                  </div>
                )}
            </div>
          </section>

          {/* Tabs des sections + Bouton Planning */}
          <div className={driverStyles.tabsContainer}>
            <div className={driverStyles.tabs}>
              <button
                className={`${driverStyles.tab} ${
                  activeTab === 'drivers' ? driverStyles.active : ''
                }`}
                onClick={() => setActiveTab('drivers')}
              >
                <span>👥 Liste des Chauffeurs</span>
                <span className={driverStyles.tabBadge}>{filteredDrivers.length}</span>
              </button>
              <button
                className={`${driverStyles.tab} ${
                  activeTab === 'hours' ? driverStyles.active : ''
                }`}
                onClick={() => setActiveTab('hours')}
              >
                <span>⏰ Heures effectuées</span>
                <span className={driverStyles.tabBadge}>{drivers.length}</span>
              </button>
            </div>
            <Link
              to={`/dashboard/company/${user?.public_id || user?.id}/driver/planning`}
              className={driverStyles.planningButton}
            >
              <span className={driverStyles.buttonIcon}>📅</span>
              Voir le Planning
            </Link>
          </div>

          {/* Tableau des Chauffeurs */}
          {activeTab === 'drivers' && (
            <>
              {loading ? (
                <div className={driverStyles.loadingState}>
                  <div className={driverStyles.spinner}></div>
                  <p>Chargement des chauffeurs...</p>
                </div>
              ) : filteredDrivers.length === 0 ? (
                <div className={driverStyles.emptyState}>
                  <p>📋 Aucun chauffeur trouvé</p>
                  <button onClick={() => setShowAddDriverModal(true)}>
                    Ajouter votre premier chauffeur
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

          {/* Tableau des heures */}
          {activeTab === 'hours' && <DriverWorkingHoursTable driverHoursData={driverHoursData} />}

          {showAddDriverModal && (
            <div className="modal-overlay" onClick={() => setShowAddDriverModal(false)}>
              <div className="modal-content modal-lg" onClick={(e) => e.stopPropagation()}>
                <h3>Ajouter un nouveau chauffeur</h3>
                <AddDriverForm
                  onSubmit={handleAddSubmit}
                  onClose={() => setShowAddDriverModal(false)}
                />
              </div>
            </div>
          )}

          {showEditDriverModal && driverToEdit && (
            <div className="modal-overlay" onClick={() => setShowEditDriverModal(false)}>
              <div className="modal-content modal-xl" onClick={(e) => e.stopPropagation()}>
                <h3>Modifier le chauffeur {driverToEdit.username}</h3>
                <EditDriverForm
                  driver={driverToEdit}
                  onSubmit={handleEditSubmit}
                  onClose={() => setShowEditDriverModal(false)}
                />
              </div>
            </div>
          )}

          <ConfirmationModal
            isOpen={showConfirmModal}
            onClose={() => setShowConfirmModal(false)}
            onConfirm={handleConfirmDelete}
            title="Confirmer la suppression"
            message={`Êtes-vous sûr de vouloir supprimer définitivement le chauffeur ${driverToDelete?.username} ?`}
          />
        </main>
      </div>
    </div>
  );
};

export default CompanyDriver;
