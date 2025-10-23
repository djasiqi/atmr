// frontend/src/pages/company/Dispatch/UnifiedDispatchRefactored.jsx
/**
 * 📊 PAGE UNIFIÉE : DISPATCH & PLANIFICATION (Version refactorisée)
 *
 * S'adapte automatiquement selon le mode configuré :
 * - MANUAL : Interface simple pour assignation manuelle
 * - SEMI_AUTO : Interface avec suggestions à valider
 * - FULLY_AUTO : Interface de surveillance avec journal d'activité
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Toaster } from 'react-hot-toast';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import useCompanySocket from '../../../hooks/useCompanySocket';
import useDispatchStatus from '../../../hooks/useDispatchStatus';
import useCompanyData from '../../../hooks/useCompanyData';

// Hooks personnalisés
import { useDispatchData } from '../../../hooks/useDispatchData';
import { useLiveDelays } from '../../../hooks/useLiveDelays';
import { useDispatchMode } from '../../../hooks/useDispatchMode';
import { useAssignmentActions } from '../../../hooks/useAssignmentActions';

// Services
import { runDispatchForDay } from '../../../services/companyService';
import {
  getOptimizerStatus,
  startRealTimeOptimizer,
  stopRealTimeOptimizer,
  applySuggestion,
} from '../../../services/dispatchMonitoringService';
import { showSuccess, showError } from '../../../utils/toast';

// Composants
import DispatchHeader from './components/DispatchHeader';
import ManualModePanel from './components/ManualModePanel';
import SemiAutoPanel from './components/SemiAutoPanel';
import FullyAutoPanel from './components/FullyAutoPanel';
import AdvancedSettings from './components/AdvancedSettings';

// Import dynamique des styles par mode
import commonStyles from './modes/Common.module.css';
import manualStyles from './modes/Manual.module.css';
import semiAutoStyles from './modes/SemiAuto.module.css';
import fullyAutoStyles from './modes/FullyAuto.module.css';

// Fonction pour fusionner les styles selon le mode actif
const getModeStyles = (mode) => {
  const modeSpecificStyles = {
    manual: manualStyles,
    semi_auto: semiAutoStyles,
    fully_auto: fullyAutoStyles,
  };

  // Fusionner les styles communs avec les styles spécifiques au mode
  return { ...commonStyles, ...(modeSpecificStyles[mode] || semiAutoStyles) };
};

// Helpers
const makeToday = () => {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
};

const UnifiedDispatchRefactored = () => {
  // Récupérer les données de l'entreprise et les chauffeurs
  const { company: _company, driver: _driver } = useCompanyData();

  // État principal
  const [date, setDate] = useState(makeToday());
  const [regularFirst, setRegularFirst] = useState(true);
  const [allowEmergency, setAllowEmergency] = useState(true);

  // 🆕 État pour overrides (chargé depuis DB au montage)
  const [overrides, setOverrides] = useState(null);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [_loadingOverrides, setLoadingOverrides] = useState(true);

  // États pour les modals
  const [selectedReservationForAssignment, setSelectedReservationForAssignment] = useState(null);

  // État pour le tri (Mode Manuel)
  const [sortBy, setSortBy] = useState('time'); // 'time', 'client', 'status'
  const [sortOrder, setSortOrder] = useState('asc'); // 'asc', 'desc'

  // États UI
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [dispatchSuccess, setDispatchSuccess] = useState(null);

  // Hooks personnalisés
  const { dispatchMode, loadDispatchMode } = useDispatchMode();
  const {
    dispatches: allDispatches,
    loading: dispatchesLoading,
    error: dispatchesError,
    loadDispatches,
  } = useDispatchData(date, dispatchMode);
  const { delays, summary: _summary, loadDelays } = useLiveDelays(date);
  const {
    handleAssignDriver,
    handleDeleteReservation,
    loading: _actionsLoading,
    error: actionsError,
    success: actionsSuccess,
  } = useAssignmentActions();

  // 🆕 Filtrer les courses CANCELED (ne pas les afficher dans le tableau)
  const dispatches = useMemo(() => {
    return (allDispatches || []).filter((d) => d.status !== 'canceled');
  }, [allDispatches]);

  // États pour l'optimiseur
  const [optimizerStatus, setOptimizerStatus] = useState(null);

  // ✅ Styles dynamiques selon le mode actif (avec fallback si mode pas encore chargé)
  const styles = getModeStyles(dispatchMode || 'semi_auto');

  // WebSocket pour temps réel
  const socket = useCompanySocket();
  const {
    label: dispatchLabel,
    progress: dispatchProgress,
    isRunning: isDispatching,
  } = useDispatchStatus(socket);

  // Charger le statut de l'optimiseur
  const loadOptimizerStatus = useCallback(async () => {
    try {
      const status = await getOptimizerStatus();
      setOptimizerStatus(status);
    } catch (err) {
      console.error('[UnifiedDispatch] Error loading optimizer:', err);
    }
  }, []);

  // Gérer l'assignation manuelle d'un chauffeur
  const onAssignDriver = async (reservationId, driverId) => {
    const success = await handleAssignDriver(reservationId, driverId);
    if (success) {
      setSelectedReservationForAssignment(null);
      loadDispatches(); // Recharger les données
    }
  };

  // Gérer la suppression d'une réservation
  const onDeleteReservation = async (reservationIdOrObject) => {
    // Extraire l'ID (peut être un objet ou un ID direct)
    const reservationId =
      typeof reservationIdOrObject === 'object' ? reservationIdOrObject.id : reservationIdOrObject;

    const success = await handleDeleteReservation(reservationId);
    if (success) {
      loadDispatches(); // Recharger les données
    }
  };

  // Lancer le dispatch
  const onRunDispatch = async () => {
    try {
      setDispatchSuccess(null);

      // ✅ FORCER allow_emergency selon overrides
      const finalAllowEmergency =
        overrides?.allow_emergency !== undefined ? overrides.allow_emergency : allowEmergency;

      console.log('🚀 [Dispatch] Lancement avec paramètres:', {
        date,
        regularFirst,
        allowEmergency: finalAllowEmergency,
        mode: dispatchMode,
        overrides: overrides,
        hasOverrides: !!overrides && Object.keys(overrides).length > 0,
      });

      const result = await runDispatchForDay({
        forDate: date,
        regularFirst: regularFirst,
        allowEmergency: finalAllowEmergency, // ✅ Utiliser override si présent
        mode: dispatchMode,
        overrides: overrides, // 🆕 Overrides personnalisés
      });

      console.log('✅ [Dispatch] Résultat reçu:', result);

      // 🔄 Rafraîchir immédiatement le tableau (ne pas attendre le WebSocket)
      setTimeout(() => {
        console.log('🔄 [Dispatch] Rafraîchissement du tableau...');
        loadDispatches();
        loadDelays();
      }, 1000); // Petit délai pour laisser le temps au worker de commit

      // ✅ Vérifier s'il y a des erreurs de validation
      if (result?.validation?.has_errors) {
        const errors = result.validation.errors || [];
        const warnings = result.validation.warnings || [];

        // Afficher message détaillé
        let message = '⚠️ Dispatch créé avec des conflits temporels !\n\n';

        if (errors.length > 0) {
          message += '🔴 ERREURS CRITIQUES :\n';
          errors.forEach((err, idx) => {
            message += `  ${idx + 1}. ${err}\n`;
          });
        }

        if (warnings.length > 0) {
          message += '\n⚠️ AVERTISSEMENTS :\n';
          warnings.forEach((warn, idx) => {
            message += `  ${idx + 1}. ${warn}\n`;
          });
        }

        message += '\n💡 Vérifiez les assignations et réassignez manuellement si nécessaire.';

        showError(message);
        setDispatchSuccess(null);
      } else if (result?.validation?.warnings) {
        // Warnings seulement (pas d'erreurs)
        showSuccess(
          '🚀 Dispatch lancé avec succès !\n⚠️ Quelques avertissements détectés (voir logs)'
        );
        setDispatchSuccess('Dispatch lancé avec avertissements');
        setTimeout(() => setDispatchSuccess(null), 5000);
      } else {
        // Succès complet sans problème
        showSuccess('🚀 Dispatch lancé avec succès !');
        setDispatchSuccess('Dispatch lancé avec succès');
        setTimeout(() => setDispatchSuccess(null), 5000);
      }
    } catch (err) {
      console.error('[UnifiedDispatch] Error running dispatch:', err);
      showError('Erreur lors du lancement du dispatch');
    }
  };

  // Charger les paramètres avancés depuis la DB au montage
  useEffect(() => {
    const loadAdvancedSettings = async () => {
      console.log('🔍 [Dispatch] Début chargement paramètres avancés...');
      try {
        const apiClient = (await import('../../../utils/apiClient')).default;
        console.log('✅ [Dispatch] apiClient chargé, appel API en cours...');
        const { data } = await apiClient.get('/company_dispatch/advanced_settings');
        console.log('📦 [Dispatch] Réponse API reçue:', data);

        if (data.dispatch_overrides) {
          setOverrides(data.dispatch_overrides);
          console.log(
            '🔄 [Dispatch] Paramètres avancés chargés depuis la DB:',
            data.dispatch_overrides
          );
        } else {
          console.log(
            '📌 [Dispatch] Aucun paramètre avancé configuré (utilise valeurs par défaut)'
          );
        }
      } catch (err) {
        console.error('❌ [Dispatch] Erreur chargement paramètres avancés:', err);
        console.error('❌ [Dispatch] Détails erreur:', err.response?.status, err.response?.data);
      } finally {
        setLoadingOverrides(false);
        console.log('✅ [Dispatch] Chargement paramètres terminé');
      }
    };

    loadAdvancedSettings();
  }, []);

  // 🆕 Handler pour appliquer overrides (temporaire, pour ce dispatch uniquement)
  const handleApplyOverrides = (newOverrides) => {
    console.log('🎯 [Overrides] Paramètres avancés appliqués (temporaire):', newOverrides);
    setOverrides(newOverrides);
    setShowAdvancedSettings(false);
    showSuccess(
      '✅ Paramètres appliqués temporairement ! Pour une sauvegarde permanente, allez dans Paramètres → Opérations.'
    );
  };

  // Gérer l'optimiseur
  const onStartOptimizer = async () => {
    try {
      await startRealTimeOptimizer();
      showSuccess('✅ Optimiseur démarré avec succès');
      loadOptimizerStatus();
    } catch (err) {
      console.error('[UnifiedDispatch] Error starting optimizer:', err);
      showError("Erreur lors du démarrage de l'optimiseur");
    }
  };

  const onStopOptimizer = async () => {
    try {
      await stopRealTimeOptimizer();
      showSuccess('⏸️ Optimiseur arrêté');
      loadOptimizerStatus();
    } catch (err) {
      console.error('[UnifiedDispatch] Error stopping optimizer:', err);
      showError("Erreur lors de l'arrêt de l'optimiseur");
    }
  };

  // Appliquer une suggestion
  const onApplySuggestion = async (suggestion) => {
    try {
      await applySuggestion(suggestion);
      showSuccess('✅ Suggestion appliquée avec succès');
      loadDispatches();
      loadDelays();
    } catch (err) {
      console.error('[UnifiedDispatch] Error applying suggestion:', err);
      showError("Erreur lors de l'application de la suggestion");
    }
  };

  // Chargement initial
  useEffect(() => {
    loadDispatches();
    loadDelays();
    loadOptimizerStatus();
    loadDispatchMode();
  }, [loadDispatches, loadDelays, loadOptimizerStatus, loadDispatchMode]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      loadDelays();
      loadOptimizerStatus();
    }, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadDelays, loadOptimizerStatus]);

  // Écoute WebSocket
  useEffect(() => {
    if (!socket) return;

    const handleDispatchComplete = (data) => {
      setDispatchSuccess(`✅ Dispatch terminé ! ${data?.assignments_count || 0} courses assignées`);
      setTimeout(() => setDispatchSuccess(null), 5000);
      loadDispatches();
      loadDelays();
    };

    const handleBookingUpdated = () => {
      loadDispatches();
      loadDelays();
    };

    socket.on('dispatch_run_completed', handleDispatchComplete);
    socket.on('booking_updated', handleBookingUpdated);
    socket.on('new_booking', handleBookingUpdated);

    return () => {
      socket.off('dispatch_run_completed', handleDispatchComplete);
      socket.off('booking_updated', handleBookingUpdated);
      socket.off('new_booking', handleBookingUpdated);
    };
  }, [socket, loadDispatches, loadDelays]);

  // Rendu du panneau selon le mode
  const renderModePanel = () => {
    const commonProps = {
      dispatches: dispatches || [],
      delays: delays || [],
      loading: dispatchesLoading,
      error: dispatchesError,
      styles,
    };

    switch (dispatchMode) {
      case 'manual':
        return (
          <ManualModePanel
            {...commonProps}
            sortBy={sortBy}
            setSortBy={setSortBy}
            sortOrder={sortOrder}
            setSortOrder={setSortOrder}
            selectedReservationForAssignment={selectedReservationForAssignment}
            setSelectedReservationForAssignment={setSelectedReservationForAssignment}
            onAssignDriver={onAssignDriver}
            onDeleteReservation={onDeleteReservation}
            currentDate={date}
          />
        );
      case 'semi_auto':
        return (
          <SemiAutoPanel
            {...commonProps}
            onApplySuggestion={onApplySuggestion}
            onDeleteReservation={onDeleteReservation}
            currentDate={date}
          />
        );
      case 'fully_auto':
        return (
          <FullyAutoPanel
            {...commonProps}
            optimizerStatus={optimizerStatus}
            onStartOptimizer={onStartOptimizer}
            onStopOptimizer={onStopOptimizer}
            autoRefresh={autoRefresh}
            setAutoRefresh={setAutoRefresh}
          />
        );
      default:
        return <div>Mode non reconnu: {dispatchMode}</div>;
    }
  };

  return (
    <div className={styles.container}>
      {/* Toast notifications provider */}
      <Toaster />

      <CompanyHeader />
      <div className={styles.mainContent}>
        <CompanySidebar />
        <div className={styles.content}>
          <DispatchHeader
            date={date}
            setDate={setDate}
            regularFirst={regularFirst}
            setRegularFirst={setRegularFirst}
            allowEmergency={allowEmergency}
            setAllowEmergency={setAllowEmergency}
            onRunDispatch={onRunDispatch}
            loading={isDispatching}
            dispatchSuccess={dispatchSuccess}
            dispatchProgress={dispatchProgress}
            dispatchLabel={dispatchLabel}
            dispatchMode={dispatchMode}
            styles={styles}
            onShowAdvancedSettings={() => setShowAdvancedSettings(true)} // 🆕
            hasOverrides={overrides !== null} // 🆕
          />

          {/* 🆕 Panneau paramètres avancés */}
          {showAdvancedSettings && (
            <div className="modal-overlay" onClick={() => setShowAdvancedSettings(false)}>
              <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <button className="modal-close" onClick={() => setShowAdvancedSettings(false)}>
                  ✕
                </button>
                <AdvancedSettings
                  key={JSON.stringify(overrides)} // 🆕 Force remount si overrides change
                  onApply={handleApplyOverrides}
                  initialSettings={overrides || {}}
                />
              </div>
            </div>
          )}

          {renderModePanel()}

          {/* Messages d'erreur/succès des actions */}
          {actionsError && <div className={styles.errorMessage}>{actionsError}</div>}
          {actionsSuccess && <div className={styles.successMessage}>{actionsSuccess}</div>}
        </div>
      </div>
    </div>
  );
};

export default UnifiedDispatchRefactored;
