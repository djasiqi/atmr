// frontend/src/hooks/useSidebarPreferences.js
// ✅ Hook pour gérer les préférences des sidebars avec sessionStorage
// Utilise sessionStorage au lieu de localStorage pour des préférences de session

import { useState, useEffect, useCallback } from 'react';

/**
 * Hook pour gérer les préférences d'expansion des sidebars
 * @param {string} sidebarType - Type de sidebar ('driver', 'admin', 'company')
 * @param {boolean} defaultValue - Valeur par défaut (true = expanded, false = collapsed)
 * @returns {[boolean, function]} - [isExpanded, setIsExpanded]
 */
const useSidebarPreferences = (sidebarType, defaultValue = true) => {
  const storageKey = `${sidebarType}SidebarExpanded`;

  // Initialiser l'état depuis sessionStorage ou valeur par défaut
  const [isExpanded, setIsExpandedState] = useState(() => {
    try {
      const saved = sessionStorage.getItem(storageKey);
      return saved !== null ? saved === 'true' : defaultValue;
    } catch (error) {
      console.warn(`Erreur lors de la lecture de sessionStorage pour ${storageKey}:`, error);
      return defaultValue;
    }
  });

  // Sauvegarder dans sessionStorage à chaque changement
  useEffect(() => {
    try {
      sessionStorage.setItem(storageKey, String(isExpanded));
    } catch (error) {
      console.warn(`Erreur lors de l'écriture dans sessionStorage pour ${storageKey}:`, error);
    }
  }, [isExpanded, storageKey]);

  // Fonction pour changer l'état
  const setIsExpanded = useCallback((value) => {
    setIsExpandedState(typeof value === 'function' ? value : () => value);
  }, []);

  return [isExpanded, setIsExpanded];
};

export default useSidebarPreferences;

