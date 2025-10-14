// src/pages/company/Driver/CompanyDriver.jsx
import React, { useEffect, useState } from "react";
import styles from "../Dashboard/CompanyDashboard.module.css";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import DriverLiveMap from "../Dashboard/components/DriverLiveMap";
import useDriver from "../../../hooks/useDriver";
import CompanyDriverTable from "../components/CompanyDriverTable";
import CompanyDriverFilters from "../components/CompanyDriverFilters";
import AddDriverForm from "../components/AddDriverForm";
import EditDriverForm from "../components/EditDriverForm";
import Modal from "../../../components/common/Modal";
import ConfirmationModal from "../../../components/common/ConfirmationModal";
import { Link } from "react-router-dom";
import {
  fetchDriverCompletedTrips,
  createDriver,
  updateDriverDetails,
} from "../../../services/companyService";
import DriverWorkingHoursTable from "./DriverWorkingHoursTable";

const CompanyDriver = () => {
  const {
    drivers,
    loading,
    error,
    toggleDriverStatus,
    deleteDriverById,
    refreshDrivers,
  } = useDriver();

  const [showAddDriverModal, setShowAddDriverModal] = useState(false);
  const [showEditDriverModal, setShowEditDriverModal] = useState(false);
  const [driverToEdit, setDriverToEdit] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [driverToDelete, setDriverToDelete] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [driverHoursData, setDriverHoursData] = useState([]);

  const filteredDrivers = (drivers || []).filter((drv) => {
    const matchesSearch = `${drv.username} ${drv.first_name} ${drv.last_name}`
      .toLowerCase()
      .includes(searchTerm.toLowerCase());
    const matchesStatus =
      statusFilter === "all" ||
      (statusFilter === "active" && drv.is_active) ||
      (statusFilter === "inactive" && !drv.is_active);
    return matchesSearch && matchesStatus;
  });

  // 3. Le useEffect utilise maintenant "drivers" pour calculer les stats
  useEffect(() => {
    async function loadStats() {
      const mappedData = [];
      for (const drv of drivers) {
        // <-- Utilise "drivers"
        let trips = [];
        try {
          trips = await fetchDriverCompletedTrips(drv.id);
        } catch (e) {
          trips = [];
        }
        const count = trips.length;
        const totalMinutes = trips.reduce(
          (sum, trip) => sum + (trip.duration_in_minutes || 0),
          0
        );
        mappedData.push({
          driverId: drv.id,
          driverName: drv.username,
          count,
          totalMinutes,
        });
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
      alert("Chauffeur ajouté avec succès !");
    } catch (err) {
      console.error("Erreur lors de l'ajout du chauffeur:", err);
      alert(`Erreur : ${err.error || "Veuillez réessayer."}`);
    }
  };

  const handleEditSubmit = async (driverId, payload) => {
    try {
      await updateDriverDetails(driverId, payload);
      refreshDrivers();
      setShowEditDriverModal(false);
      alert("Chauffeur mis à jour avec succès !");
    } catch (err) {
      console.error("Erreur de mise à jour:", err);
      alert(`Erreur : ${err.error || "Veuillez réessayer."}`);
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

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {error && <div className={styles.error}>{error}</div>}
          <div className={styles.headerActions}>
            <button
              className={styles.addButton}
              onClick={() => setShowAddDriverModal(true)}
            >
              Ajouter Chauffeur
            </button>
          </div>

          <section className={styles.mapSection}>
            <h2>Localisation des Chauffeurs</h2>
            <DriverLiveMap />
          </section>

          <CompanyDriverFilters
            searchTerm={searchTerm}
            setSearchTerm={setSearchTerm}
            statusFilter={statusFilter}
            setStatusFilter={setStatusFilter}
          />

          {loading ? (
            <div className={styles.loading}>Chargement...</div>
          ) : filteredDrivers.length === 0 ? (
            <p>Aucun chauffeur trouvé.</p>
          ) : (
            // 5. Le tableau reçoit les bonnes props
            <CompanyDriverTable
              drivers={filteredDrivers}
              onEdit={openEditModal}
              onToggleStatus={toggleDriverStatus}
              onDeleteRequest={handleDeleteRequest}
            />
          )}

          {/* Section affichant le tableau d'heures réelles */}
          <section className={styles.workingHoursSection}>
            <h2>Tableau des heures effectuées</h2>
            <DriverWorkingHoursTable driverHoursData={driverHoursData} />
          </section>

          <section>
            <Link to="/company/driver/planning">Voir le Planning</Link>
          </section>

          {showAddDriverModal && (
            <Modal onClose={() => setShowAddDriverModal(false)}>
              <h3>Ajouter un nouveau chauffeur</h3>
              <AddDriverForm
                onSubmit={handleAddSubmit}
                onClose={() => setShowAddDriverModal(false)}
              />
            </Modal>
          )}

          {showEditDriverModal && driverToEdit && (
            <Modal onClose={() => setShowEditDriverModal(false)}>
              <h3>Modifier le chauffeur {driverToEdit.username}</h3>
              <EditDriverForm
                driver={driverToEdit}
                onSubmit={handleEditSubmit}
                onClose={() => setShowEditDriverModal(false)}
              />
            </Modal>
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
