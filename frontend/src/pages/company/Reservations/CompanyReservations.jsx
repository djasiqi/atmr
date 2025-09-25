import React, {
  useEffect,
  useState,
  useCallback,
  useMemo,
  useRef,
} from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import { fetchCompanyReservations, deleteReservation } from "../../../services/companyService";
import ReservationTable from "../Dashboard/components/ReservationTable";
import ReservationDetailsModal from "../Dashboard/components/ReservationDetailsModal";
import ConfirmationModal from "../../../components/common/ConfirmationModal";
import styles from "./CompanyReservations.module.css";

const CompanyReservations = () => {
  const [reservations, setReservations] = useState([]);
  const [filteredReservations, setFilteredReservations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortOrder, setSortOrder] = useState("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const reservationsPerPage = 10;
  const [selectedReservation, setSelectedReservation] = useState(null);

  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [reservationToDelete, setReservationToDelete] = useState(null);

  const didShowToast = useRef(false);

  // Le chargement des réservations reste inchangé...
  const loadReservations = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchCompanyReservations();
      // Ensure we're handling both array and {reservations: [...]} formats
      setReservations(Array.isArray(data) ? data : (data.reservations || []));
      if (!didShowToast.current) {
        toast.success("Réservations chargées !");
        didShowToast.current = true;
      }
    } catch (err) {
      console.error("Erreur lors du chargement des réservations :", err);
      setError("Une erreur est survenue lors du chargement des réservations.");
      toast.error("Erreur lors du chargement des réservations.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadReservations();
  }, [loadReservations]);

  // Dans le composant CompanyReservations

  const handleDeleteRequest = (reservation) => {
    setReservationToDelete(reservation);
    setShowConfirmModal(true);
  };

    const handleCloseConfirmModal = () => {
    setShowConfirmModal(false);
    setReservationToDelete(null);
  };

  const handleConfirmDelete = async () => {
    if (!reservationToDelete) return;
    try {
      const response = await deleteReservation(reservationToDelete.id);
      toast.success(response.message || "Réservation supprimée !");
      setReservations(prev => prev.filter(r => r.id !== reservationToDelete.id));
    } catch (err) {
      console.error("Erreur lors de la suppression:", err);
      toast.error(err.error || "Une erreur est survenue.");
    } finally {
      handleCloseConfirmModal();
    }
  };



  // Filtrer et trier les réservations
  useEffect(() => {
    let filtered = [...reservations];
    if (searchTerm) {
      filtered = filtered.filter((r) =>
        r.customer_name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    if (statusFilter !== "all") {
      filtered = filtered.filter(
        (r) => r.status.toLowerCase() === statusFilter
      );
    }
    filtered.sort((a, b) => {
      const dateA = new Date(a.scheduled_time);
      const dateB = new Date(b.scheduled_time);
      return sortOrder === "asc" ? dateA - dateB : dateB - dateA;
    });
    setFilteredReservations(filtered);
    setCurrentPage(1);
  }, [reservations, searchTerm, statusFilter, sortOrder]);

  // Pagination
  const currentReservations = useMemo(() => {
    const indexOfLast = currentPage * reservationsPerPage;
    const indexOfFirst = indexOfLast - reservationsPerPage;
    return filteredReservations.slice(indexOfFirst, indexOfLast);
  }, [filteredReservations, currentPage]);

  const totalPages = Math.ceil(
    filteredReservations.length / reservationsPerPage
  );

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {error && <div className={styles.error}>{error}</div>}
          <div className={styles.filters}>
            <input
              type="text"
              placeholder="Rechercher par client..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={styles.searchInput}
            />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className={styles.statusFilter}
            >
              <option value="all">Tous</option>
              <option value="pending">En attente</option>
              <option value="assigned">Assignée</option>
              <option value="completed">Terminée</option>
              <option value="canceled">Annulée</option>
            </select>
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value)}
              className={styles.sortSelect}
            >
              <option value="desc">Plus récent</option>
              <option value="asc">Plus ancien</option>
            </select>
          </div>

          {loading ? (
            <p>Chargement des réservations...</p>
          ) : filteredReservations.length === 0 ? (
            <p>Aucune réservation trouvée.</p>
          ) : (
            <>
              <ReservationTable
                reservations={currentReservations}
                onRowClick={(reservation) => setSelectedReservation(reservation)}
                onDelete={handleDeleteRequest} // On passe la fonction qui OUVRE le modal
              />
              <div className={styles.pagination}>
                <button
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage(currentPage - 1)}
                >
                  Précédent
                </button>
                <span>
                  Page {currentPage} sur {totalPages}
                </span>
                <button
                  disabled={currentPage === totalPages}
                  onClick={() => setCurrentPage(currentPage + 1)}
                >
                  Suivant
                </button>
              </div>
            </>
          )}

          {selectedReservation && (
            <ReservationDetailsModal
              reservation={selectedReservation}
              onClose={() => setSelectedReservation(null)}
            />
          )}

          <ConfirmationModal
            isOpen={showConfirmModal}
            onClose={handleCloseConfirmModal}
            onConfirm={handleConfirmDelete}
            title={`Supprimer la Réservation n°${reservationToDelete?.id}`}
            confirmText="Oui, supprimer"
          >
            <p>
                Êtes-vous sûr de vouloir supprimer la réservation pour <strong>{reservationToDelete?.customer_name}</strong> ?
            </p>
            <p style={{color: '#ef4444', fontStyle: 'italic', marginTop: '16px'}}>
                Cette action est irréversible.
            </p>
          </ConfirmationModal>

        </main>
      </div>
      <ToastContainer position="top-right" autoClose={3000} />
    </div>
  );
};

export default CompanyReservations;
