// src/hooks/useDispatchStatus.js
import { useState, useEffect, useCallback } from 'react';
import { getDispatchStatus, fetchAssignedReservations } from '../services/companyService';

export default function useDispatchStatus(socket) {
  const [status, setStatus] = useState('idle'); // idle, running, completed, failed
  const [progress, setProgress] = useState(0);
  const [label, setLabel] = useState('Prêt');
  const [lastRunId, setLastRunId] = useState(null);
  const [updatedAt, setUpdatedAt] = useState(Date.now());
  const [pollingActive, setPollingActive] = useState(false);
  const [currentDate, setCurrentDate] = useState(null);

  // Polling du statut (backup si pas de socket)
  useEffect(() => {
    if (!pollingActive) return;

    const fetchStatus = async () => {
      try {
        const data = await getDispatchStatus();
        const isRunning = data?.is_running || false;
        setStatus(isRunning ? 'running' : 'idle');
        setProgress(data?.progress || 0);
        setLabel(isRunning ? `Optimisation en cours (${data?.progress || 0}%)` : 'Prêt');

        // Si on atteint 100% ou si on n'est plus en cours, on arrête le polling
        if (data?.progress >= 100 || !isRunning) {
          setPollingActive(false);
          setStatus('completed');
          setProgress(100);
          setLabel('Optimisation terminée');
          setUpdatedAt(Date.now());
        }
      } catch (error) {
        console.error('Erreur lors de la récupération du statut:', error);
        setPollingActive(false);
      }
    };

    const intervalId = setInterval(fetchStatus, 2000);
    return () => clearInterval(intervalId);
  }, [pollingActive, setStatus, setProgress, setLabel, setPollingActive, setUpdatedAt]);

  // Fonction pour rafraîchir les assignations pour une date donnée
  const refreshAssignmentsForDate = useCallback(async (dateStr) => {
    try {
      // Attendre un court instant pour s'assurer que les données sont persistées
      await new Promise((resolve) => setTimeout(resolve, 500));
      // Récupérer les assignations pour cette date
      const assignments = await fetchAssignedReservations(dateStr);
      return assignments;
    } catch (error) {
      console.error(`❌ Error refreshing assignments for date ${dateStr}:`, error);
      throw error;
    }
  }, []);

  // Écoute des événements socket
  useEffect(() => {
    if (!socket) return;

    const handleDispatchStatus = (data) => {
      setStatus(data.is_running ? 'running' : 'idle');
      setProgress(data.progress || 0);
      setLabel(data.is_running ? `Optimisation en cours (${data.progress || 0}%)` : 'Prêt');

      if (!data.is_running) {
        setUpdatedAt(Date.now());
      }
    };

    const handleDispatchRunCompleted = (data) => {
      setStatus('completed');
      setProgress(100);
      setLabel('Optimisation terminée');
      setLastRunId(data.dispatch_run_id);
      setPollingActive(false);
      setUpdatedAt(Date.now());

      // Si la date est fournie dans l'événement, la mémoriser pour le rafraîchissement
      if (data.date) {
        setCurrentDate(data.date);
        // Déclencher un rafraîchissement des données pour cette date
        refreshAssignmentsForDate(data.date);
      } else if (currentDate) {
        // Try to refresh with the last known date if available
        refreshAssignmentsForDate(currentDate);
      }
    };

    socket.on('dispatch_status', handleDispatchStatus);
    socket.on('dispatch_run_completed', handleDispatchRunCompleted);

    return () => {
      socket.off('dispatch_status', handleDispatchStatus);
      socket.off('dispatch_run_completed', handleDispatchRunCompleted);
    };
  }, [socket, refreshAssignmentsForDate, currentDate]);

  // Traitement de la réponse d'un job de dispatch
  const handleDispatchJobResponse = useCallback(
    (response) => {
      // Si on a un job_id, c'est un job async
      if (response?.job_id) {
        setStatus('running');
        setProgress(5); // Valeur initiale
        setLabel('Optimisation en cours (5%)');
        setPollingActive(true); // Active le polling

        // Mémoriser la date si elle est présente dans la réponse
        if (response.for_date) {
          setCurrentDate(response.for_date);
        }

        // Mémoriser le dispatch_run_id si présent
        if (response.dispatch_run_id) {
          setLastRunId(response.dispatch_run_id);
        }
      }
      // Si on a un résultat direct (sync)
      else if (response?.assignments) {
        setStatus('completed');
        setProgress(100);
        setLabel('Optimisation terminée');
        setLastRunId(response.dispatch_run_id || response.meta?.dispatch_run_id);
        setUpdatedAt(Date.now());

        // Mémoriser la date si elle est présente dans la réponse
        if (response.for_date) {
          setCurrentDate(response.for_date);
          // Rafraîchir les assignations immédiatement
          refreshAssignmentsForDate(response.for_date);
        }
      }
    },
    [refreshAssignmentsForDate]
  );

  return {
    status,
    progress,
    label,
    isRunning: status === 'running',
    lastRunId,
    updatedAt,
    currentDate,
    setUpdatedAt,
    handleDispatchJobResponse,
    refreshAssignmentsForDate,
  };
}
