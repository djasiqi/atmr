// src/pages/Dashboard/DriverDashboard.jsx
import React, { useEffect, useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import DriverSidebar from '../../../components/layout/Sidebar/DriverSidebar/DriverSidebar';
import DriverProfile from '../Profile/DriverProfile';
import DriverProfileModal from '../components/Profile/DriverProfileModal';
import DriverOverview from '../components/Dashboard/DriverOverview';
import DriverSchedule from '../components/Dashboard/DriverSchedule';
import CourseList from '../components/Dashboard/CourseList';
import CourseDetailsModal from '../../../components/widgets/CourseDetailsModal';
import DriverMap from '../components/Dashboard/DriverMap';
import Notifications from '../components/Dashboard/Notifications';
import DriverHistory from '../components/Dashboard/DriverHistory';
import DriverReports from '../components/Dashboard/DriverReports';
import VoiceControl from '../components/Dashboard/VoiceControl';
import useDriverLocation from '../../../hooks/useDriverLocation';
import styles from './DriverDashboard.module.css';

const DriverDashboard = () => {
  const { public_id } = useParams();
  const [assignments, setAssignments] = useState([]);
  const [, setLoading] = useState(true);
  const [, setError] = useState(null);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [reports, setReports] = useState({
    totalCourses: 0,
    totalTime: 0,
    totalRevenue: 0,
  });
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [profileToEdit, setProfileToEdit] = useState(null);

  // Utilisation du hook pour obtenir la position GPS du chauffeur
  const { location: myLocation, error: locationError } = useDriverLocation(
    (newLocation) => {
      console.log('Position mise à jour:', newLocation);
      // Ici, vous pouvez envoyer la nouvelle position à votre serveur en temps réel via Socket.IO
    },
    { enableHighAccuracy: true, maximumAge: 5000, timeout: 10000 }
  );

  const fetchDriverData = useCallback(async () => {
    try {
      setLoading(true);
      // Exemple de données simulées
      const fetchedAssignments = [
        {
          id: 1,
          customer_name: 'Client A',
          scheduled_time: new Date().toISOString(),
          pickup: 'Adresse A',
          dropoff: 'Adresse B',
          status: 'assigned',
          revenue: 50,
          location: { lat: '46.2044', lng: '6.1432' },
          is_on_trip: false,
          route: [],
        },
      ];
      setAssignments(fetchedAssignments);
      setReports({
        totalCourses: fetchedAssignments.length,
        totalTime: 30,
        totalRevenue: 50,
      });
    } catch (err) {
      console.error(err);
      setError('Erreur lors du chargement des données du chauffeur.');
    } finally {
      setLoading(false);
    }
  }, []);

  const closeModal = () => {
    setSelectedCourse(null);
  };

  useEffect(() => {
    fetchDriverData();
  }, [fetchDriverData]);

  return (
    <div className={styles.driverDashboard}>
      <HeaderDashboard />
      <DriverSidebar />
      <main className={styles.mainContent}>
        {/* Affichage du profil avec possibilité d'édition */}
        <DriverProfile
          public_id={public_id}
          onEdit={(profile) => {
            setProfileToEdit(profile);
            setShowProfileModal(true);
          }}
        />

        {/* Notifications globales */}
        <Notifications />

        {/* Vue d'ensemble */}
        <DriverOverview assignments={assignments} />

        {/* Planning de la journée */}
        <DriverSchedule assignments={assignments} />

        {/* Liste des courses assignées */}
        <CourseList assignments={assignments} onRowClick={(course) => setSelectedCourse(course)} />

        {/* Carte en temps réel : 
            Nous passons la position actuelle du chauffeur (myLocation) en plus des assignments */}
        <DriverMap assignments={assignments} myLocation={myLocation} isLoaded={true} />

        {/* Modal des détails d'une course */}
        {selectedCourse && (
          <CourseDetailsModal
            course={selectedCourse}
            onClose={closeModal} // Utilisation de la fonction définie
            onStartCourse={(id) => {
              // Logique pour démarrer la course, par exemple appeler l'API startBooking
              console.log('Démarrer la course', id);
            }}
            onReportIssue={(id) => {
              // Logique pour signaler un problème
              console.log('Signaler un problème pour la course', id);
            }}
            onCompleteCourse={(id) => {
              // Logique pour terminer la course
              console.log('Terminer la course', id);
            }}
          />
        )}

        {/* Historique des courses terminées */}
        <DriverHistory history={assignments.filter((a) => a.status === 'completed')} />

        {/* Rapports du jour */}
        <DriverReports reports={reports} />

        {/* Commande vocale */}
        <VoiceControl />

        {/* Modal pour l'édition du profil */}
        {showProfileModal && profileToEdit && (
          <DriverProfileModal
            profile={profileToEdit}
            onClose={() => setShowProfileModal(false)}
            onSave={(updatedData) => {
              setProfileToEdit({ ...profileToEdit, ...updatedData });
              setShowProfileModal(false);
            }}
          />
        )}

        {/* Optionnel : Affichage d'erreurs de géolocalisation */}
        {locationError && (
          <div className={styles.error}>Erreur de géolocalisation : {locationError}</div>
        )}
      </main>
    </div>
  );
};

export default DriverDashboard;
