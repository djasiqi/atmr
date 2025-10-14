import React, { useState, useEffect } from "react";
import {
  fetchCompanyClients,
  createClient,
  updateClient,
  deleteClient,
} from "../../../services/companyService";
import { invoiceService } from "../../../services/invoiceService";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import ClientsTable from "./components/ClientsTable";
import EditClientModal from "./components/EditClientModal";
import NewClientModal from "./components/NewClientModal";
import DeleteConfirmModal from "./components/DeleteConfirmModal";
import styles from "./CompanyClients.module.css";

const CompanyClients = () => {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all"); // 'all', 'regular', 'institution'
  const [editingClient, setEditingClient] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showNewClientModal, setShowNewClientModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [clientToDelete, setClientToDelete] = useState(null);

  // Charger les clients
  const loadClients = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchCompanyClients();
      setClients(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error("Erreur lors du chargement des clients:", err);
      setError("Impossible de charger les clients");
      setClients([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadClients();
  }, []);

  // Filtrer les clients
  const filteredClients = clients.filter((client) => {
    // Filtre par texte
    const matchesSearch = searchTerm
      ? (client.first_name || "")
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        (client.last_name || "")
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        (client.full_name || "")
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        (client.institution_name || "")
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        (client.contact_email || "")
          .toLowerCase()
          .includes(searchTerm.toLowerCase())
      : true;

    // Filtre par type
    const matchesType =
      filterType === "all"
        ? true
        : filterType === "institution"
        ? client.is_institution
        : !client.is_institution;

    return matchesSearch && matchesType;
  });

  // Ouvrir le modal d'édition
  const handleEditClient = (client) => {
    setEditingClient(client);
    setShowEditModal(true);
  };

  // Ouvrir le modal de suppression
  const handleDeleteClick = (client) => {
    setClientToDelete(client);
    setShowDeleteModal(true);
  };

  // Fermer le modal
  const handleCloseModal = () => {
    setShowEditModal(false);
    setEditingClient(null);
  };

  // Fermer le modal de suppression
  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
    setClientToDelete(null);
  };

  // Confirmer la suppression
  const handleConfirmDelete = async (hardDelete = false) => {
    if (!clientToDelete) return;

    try {
      await deleteClient(clientToDelete.id, hardDelete);
      await loadClients();
      handleCloseDeleteModal();
    } catch (err) {
      console.error("Erreur lors de la suppression:", err);
      
      // Message d'erreur détaillé
      let errorMessage = err.error || err.message || "Erreur lors de la suppression";
      
      if (err.reason) {
        errorMessage += "\n\n" + err.reason;
      }
      
      if (err.suggestion) {
        errorMessage += "\n\n💡 " + err.suggestion;
      }
      
      alert(errorMessage);
    }
  };

  // Sauvegarder les modifications
  const handleSaveClient = async (clientData) => {
    try {
      // Utiliser l'API complète de mise à jour du client
      await updateClient(editingClient.id, clientData);

      // Recharger la liste
      await loadClients();
      handleCloseModal();
    } catch (err) {
      console.error("Erreur lors de la sauvegarde:", err);
      throw err;
    }
  };

  // Créer un nouveau client
  const handleCreateClient = async (clientData) => {
    try {
      console.log("Création client avec données:", clientData);

      const newClient = await createClient(clientData);
      console.log("Client créé:", newClient);

      // Recharger la liste complète
      await loadClients();
      setShowNewClientModal(false);
    } catch (err) {
      console.error("Erreur lors de la création du client:", err);
      console.error("Détails:", err.response?.data);
      throw err;
    }
  };

  // Statistiques
  const stats = {
    total: clients.length,
    regular: clients.filter((c) => !c.is_institution).length,
    institutions: clients.filter((c) => c.is_institution).length,
    active: clients.filter((c) => c.is_active).length,
  };

  return (
    <>
      <CompanyHeader />
      <div className={styles.layout}>
        <CompanySidebar />
        <div className={styles.container}>
          <div className={styles.header}>
            <div>
              <h1 className={styles.title}>Gestion des clients</h1>
              <p className={styles.subtitle}>
                Gérez vos clients et institutions
              </p>
            </div>
            <button
              onClick={() => setShowNewClientModal(true)}
              className={styles.addBtn}
            >
              + Ajouter un client
            </button>
          </div>

          {/* Statistiques */}
          <div className={styles.statsGrid}>
            <div className={styles.statCard}>
              <div className={styles.statValue}>{stats.total}</div>
              <div className={styles.statLabel}>Total clients</div>
            </div>
            <div className={styles.statCard}>
              <div className={styles.statValue}>{stats.regular}</div>
              <div className={styles.statLabel}>Clients réguliers</div>
            </div>
            <div className={styles.statCard}>
              <div className={styles.statValue}>{stats.institutions}</div>
              <div className={styles.statLabel}>Institutions</div>
            </div>
            <div className={styles.statCard}>
              <div className={styles.statValue}>{stats.active}</div>
              <div className={styles.statLabel}>Actifs</div>
            </div>
          </div>

          {/* Filtres */}
          <div className={styles.filters}>
            <div className={styles.searchBox}>
              <input
                type="text"
                placeholder="Rechercher un client..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className={styles.searchInput}
              />
            </div>

            <div className={styles.typeFilters}>
              <button
                className={`${styles.filterBtn} ${
                  filterType === "all" ? styles.active : ""
                }`}
                onClick={() => setFilterType("all")}
              >
                Tous ({stats.total})
              </button>
              <button
                className={`${styles.filterBtn} ${
                  filterType === "regular" ? styles.active : ""
                }`}
                onClick={() => setFilterType("regular")}
              >
                Clients ({stats.regular})
              </button>
              <button
                className={`${styles.filterBtn} ${
                  filterType === "institution" ? styles.active : ""
                }`}
                onClick={() => setFilterType("institution")}
              >
                Institutions ({stats.institutions})
              </button>
            </div>
          </div>

          {/* Contenu principal */}
          {loading && (
            <div className={styles.loading}>Chargement des clients...</div>
          )}

          {error && (
            <div className={styles.error}>
              {error}
              <button onClick={loadClients} className={styles.retryBtn}>
                Réessayer
              </button>
            </div>
          )}

          {!loading && !error && (
            <ClientsTable
              clients={filteredClients}
              onEdit={handleEditClient}
              onDelete={handleDeleteClick}
              onRefresh={loadClients}
            />
          )}

          {/* Modal d'édition */}
          {showEditModal && editingClient && (
            <EditClientModal
              client={editingClient}
              onClose={handleCloseModal}
              onSave={handleSaveClient}
            />
          )}

          {/* Modal création */}
          {showNewClientModal && (
            <NewClientModal
              onClose={() => setShowNewClientModal(false)}
              onSave={handleCreateClient}
            />
          )}

          {/* Modal suppression */}
          {showDeleteModal && clientToDelete && (
            <DeleteConfirmModal
              client={clientToDelete}
              onClose={handleCloseDeleteModal}
              onConfirm={handleConfirmDelete}
            />
          )}
        </div>
      </div>
    </>
  );
};

export default CompanyClients;
