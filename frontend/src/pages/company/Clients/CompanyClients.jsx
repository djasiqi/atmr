import React, { useState, useEffect } from 'react';
import {
  fetchCompanyClients,
  createClient,
  updateClient,
  deleteClient,
} from '../../../services/companyService';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import ClientsTable from './components/ClientsTable';
import EditClientModal from './components/EditClientModal';
import NewClientModal from './components/NewClientModal';
import DeleteConfirmModal from './components/DeleteConfirmModal';
import styles from './CompanyClients.module.css';

const CompanyClients = () => {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all', 'regular', 'institution'
  const [editingClient, setEditingClient] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showNewClientModal, setShowNewClientModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [clientToDelete, setClientToDelete] = useState(null);

  // Nouveaux états pour pagination et tri
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [sortBy, setSortBy] = useState('name'); // 'name', 'email', 'created'
  const [sortOrder, setSortOrder] = useState('asc'); // 'asc', 'desc'

  // Charger les clients
  const loadClients = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchCompanyClients();
      setClients(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Erreur lors du chargement des clients:', err);
      setError('Impossible de charger les clients');
      setClients([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadClients();
  }, []);

  // Filtrer et trier les clients
  const filteredAndSortedClients = React.useMemo(() => {
    // 1. Filtrer
    let filtered = clients.filter((client) => {
      // Filtre par texte
      const matchesSearch = searchTerm
        ? (client.first_name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
          (client.last_name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
          (client.full_name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
          (client.institution_name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
          (client.contact_email || '').toLowerCase().includes(searchTerm.toLowerCase())
        : true;

      // Filtre par type
      const matchesType =
        filterType === 'all'
          ? true
          : filterType === 'institution'
          ? client.is_institution
          : !client.is_institution;

      return matchesSearch && matchesType;
    });

    // 2. Trier
    filtered.sort((a, b) => {
      let compareA, compareB;

      switch (sortBy) {
        case 'name':
          compareA = (
            a.full_name ||
            `${a.first_name} ${a.last_name}` ||
            a.institution_name ||
            ''
          ).toLowerCase();
          compareB = (
            b.full_name ||
            `${b.first_name} ${b.last_name}` ||
            b.institution_name ||
            ''
          ).toLowerCase();
          break;
        case 'email':
          compareA = (a.contact_email || a.email || '').toLowerCase();
          compareB = (b.contact_email || b.email || '').toLowerCase();
          break;
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

    return filtered;
  }, [clients, searchTerm, filterType, sortBy, sortOrder]);

  // 3. Paginer
  const totalPages = Math.ceil(filteredAndSortedClients.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedClients = filteredAndSortedClients.slice(startIndex, endIndex);

  // Réinitialiser à la page 1 quand les filtres changent
  React.useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, filterType]);

  // Gérer le changement de page
  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      // Scroll vers le haut du tableau
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

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
      console.error('Erreur lors de la suppression:', err);

      // Message d'erreur détaillé
      let errorMessage = err.error || err.message || 'Erreur lors de la suppression';

      if (err.reason) {
        errorMessage += '\n\n' + err.reason;
      }

      if (err.suggestion) {
        errorMessage += '\n\n💡 ' + err.suggestion;
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
      console.error('Erreur lors de la sauvegarde:', err);
      throw err;
    }
  };

  // Créer un nouveau client
  const handleCreateClient = async (clientData) => {
    try {
      console.log('Création client avec données:', clientData);

      const newClient = await createClient(clientData);
      console.log('Client créé:', newClient);

      // Recharger la liste complète
      await loadClients();
      setShowNewClientModal(false);
    } catch (err) {
      console.error('Erreur lors de la création du client:', err);
      console.error('Détails:', err.response?.data);
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
          {/* Section Header + Filtres */}
          <section className={styles.headerSection}>
            <div className={styles.header}>
              <div className={styles.headerLeft}>
                <h1 className={styles.title}>Gestion des clients</h1>
                <p className={styles.subtitle}>Gérez vos clients et institutions</p>
              </div>
              <button
                onClick={() => setShowNewClientModal(true)}
                className={`btn btn-primary ${styles.addBtn}`}
              >
                ➕ Ajouter un client
              </button>
            </div>

            {/* Filtres dans le même conteneur */}
            <div className={styles.filters}>
              <div className={styles.searchBox}>
                <label className={styles.searchLabel}>🔍 Recherche globale</label>
                <input
                  type="text"
                  placeholder="ID, nom, email, téléphone, type de client..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className={styles.searchInput}
                />
              </div>

              <div className={styles.typeFilters}>
                <button
                  className={`${styles.filterBtn} ${filterType === 'all' ? styles.active : ''}`}
                  onClick={() => setFilterType('all')}
                >
                  Tous
                </button>
                <button
                  className={`${styles.filterBtn} ${filterType === 'regular' ? styles.active : ''}`}
                  onClick={() => setFilterType('regular')}
                >
                  Clients
                </button>
                <button
                  className={`${styles.filterBtn} ${
                    filterType === 'institution' ? styles.active : ''
                  }`}
                  onClick={() => setFilterType('institution')}
                >
                  Institutions
                </button>
              </div>
            </div>
          </section>

          {/* Statistiques KPI */}
          <div className={styles.statsGrid}>
            <div className={styles.statCard}>
              <span className={styles.statIcon}>👥</span>
              <div className={styles.statContent}>
                <h3 className={styles.statLabel}>Total clients</h3>
                <p className={styles.statValue}>{stats.total}</p>
              </div>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statIcon}>👤</span>
              <div className={styles.statContent}>
                <h3 className={styles.statLabel}>Clients réguliers</h3>
                <p className={styles.statValue}>{stats.regular}</p>
              </div>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statIcon}>🏢</span>
              <div className={styles.statContent}>
                <h3 className={styles.statLabel}>Institutions</h3>
                <p className={styles.statValue}>{stats.institutions}</p>
              </div>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statIcon}>✅</span>
              <div className={styles.statContent}>
                <h3 className={styles.statLabel}>Actifs</h3>
                <p className={styles.statValue}>{stats.active}</p>
              </div>
            </div>
          </div>

          {/* Contenu principal */}
          {loading && <div className={styles.loading}>Chargement des clients...</div>}

          {error && (
            <div className={styles.error}>
              {error}
              <button onClick={loadClients} className="btn btn-sm btn-danger">
                🔄 Réessayer
              </button>
            </div>
          )}

          {!loading && !error && (
            <>
              {/* Barre d'outils : tri + pagination */}
              <div className={styles.toolbar}>
                <div className={styles.toolbarLeft}>
                  <div className={styles.sortControls}>
                    <label htmlFor="sortBy">Trier par:</label>
                    <select
                      id="sortBy"
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value)}
                      className={styles.sortSelect}
                    >
                      <option value="name">Nom</option>
                      <option value="email">Email</option>
                      <option value="created">Date de création</option>
                    </select>

                    <button
                      onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                      className={styles.sortOrderBtn}
                      title={sortOrder === 'asc' ? 'Ordre croissant' : 'Ordre décroissant'}
                    >
                      {sortOrder === 'asc' ? '↑' : '↓'}
                    </button>
                  </div>
                </div>

                <div className={styles.toolbarRight}>
                  <label htmlFor="itemsPerPage">Afficher:</label>
                  <select
                    id="itemsPerPage"
                    value={itemsPerPage}
                    onChange={(e) => {
                      setItemsPerPage(Number(e.target.value));
                      setCurrentPage(1);
                    }}
                    className={styles.perPageSelect}
                  >
                    <option value={10}>10</option>
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                  </select>
                </div>
              </div>

              {/* Tableau des clients */}
              <ClientsTable
                clients={paginatedClients}
                onEdit={handleEditClient}
                onDelete={handleDeleteClick}
                onRefresh={loadClients}
              />

              {/* Pagination */}
              {totalPages > 1 && (
                <div className={styles.paginationContainer}>
                  <div className={styles.pagination}>
                    <button
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1}
                      className={styles.paginationBtn}
                    >
                      ← Précédent
                    </button>

                    <span className={styles.pageInfo}>
                      Page {currentPage} sur {totalPages}
                    </span>

                    <button
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages}
                      className={styles.paginationBtn}
                    >
                      Suivant →
                    </button>
                  </div>
                </div>
              )}
            </>
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
