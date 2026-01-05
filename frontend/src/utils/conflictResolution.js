// utils/conflictResolution.js

/**
 * Détecte si il y a un conflit entre l'état optimiste et la réponse du serveur
 * @param {*} optimisticState - État optimiste (mise à jour locale)
 * @param {*} serverState - État retourné par le serveur
 * @param {string} idField - Nom du champ ID (défaut: 'id')
 * @returns {boolean} true si conflit détecté
 */
export function detectConflict(optimisticState, serverState, idField = 'id') {
  if (!optimisticState || !serverState) return false;
  
  // Comparaison simple: si les IDs correspondent mais les données diffèrent
  if (optimisticState[idField] !== serverState[idField]) {
    return false; // Pas le même objet
  }

  // Comparer les champs critiques (exclure les timestamps et métadonnées)
  const fieldsToCompare = Object.keys(optimisticState).filter(
    key => !key.includes('_at') && 
           !key.includes('timestamp') && 
           !key.includes('updated') &&
           key !== 'version' &&
           key !== 'etag'
  );

  for (const field of fieldsToCompare) {
    if (JSON.stringify(optimisticState[field]) !== JSON.stringify(serverState[field])) {
      return true; // Conflit détecté
    }
  }

  return false;
}

/**
 * Stratégie de résolution: serveur gagne (server-wins)
 * @param {*} optimisticState - État optimiste
 * @param {*} serverState - État serveur
 * @returns {*} État final (serveur)
 */
export function resolveServerWins(optimisticState, serverState) {
  return serverState;
}

/**
 * Stratégie de résolution: client gagne (client-wins)
 * @param {*} optimisticState - État optimiste
 * @param {*} _serverState - État serveur (non utilisé)
 * @returns {*} État final (client)
 */
export function resolveClientWins(optimisticState, _serverState) {
  return optimisticState;
}

/**
 * Stratégie de résolution: merge intelligent
 * Fusionne les deux états en privilégiant le serveur pour les champs critiques
 * @param {*} optimisticState - État optimiste
 * @param {*} serverState - État serveur
 * @param {Array<string>} serverPriorityFields - Champs où le serveur a priorité
 * @returns {*} État fusionné
 */
export function resolveMerge(optimisticState, serverState, serverPriorityFields = ['status', 'driver_id', 'assignment_id']) {
  const merged = { ...optimisticState, ...serverState };
  
  // Pour les champs critiques, utiliser le serveur
  for (const field of serverPriorityFields) {
    if (serverState[field] !== undefined) {
      merged[field] = serverState[field];
    }
  }
  
  return merged;
}

/**
 * Résout un conflit selon la stratégie spécifiée
 * @param {string} strategy - 'server-wins', 'client-wins', 'merge', ou 'user-choice'
 * @param {*} optimisticState - État optimiste
 * @param {*} serverState - État serveur
 * @param {Function} userChoiceCallback - Callback pour 'user-choice' (retourne la décision)
 * @returns {Promise<*>} État résolu
 */
export async function resolveConflict(strategy, optimisticState, serverState, userChoiceCallback = null) {
  switch (strategy) {
    case 'server-wins':
      return resolveServerWins(optimisticState, serverState);
    
    case 'client-wins':
      return resolveClientWins(optimisticState, serverState);
    
    case 'merge':
      return resolveMerge(optimisticState, serverState);
    
    case 'user-choice':
      if (!userChoiceCallback) {
        console.warn('user-choice strategy requires userChoiceCallback');
        return resolveServerWins(optimisticState, serverState); // Fallback
      }
      const userChoice = await userChoiceCallback(optimisticState, serverState);
      return userChoice === 'client' 
        ? resolveClientWins(optimisticState, serverState)
        : resolveServerWins(optimisticState, serverState);
    
    default:
      console.warn(`Unknown conflict resolution strategy: ${strategy}, defaulting to server-wins`);
      return resolveServerWins(optimisticState, serverState);
  }
}

/**
 * Crée un message de conflit pour l'UI
 * @param {*} optimisticState - État optimiste
 * @param {*} serverState - État serveur
 * @returns {string} Message de conflit
 */
export function createConflictMessage(optimisticState, serverState) {
  const differences = [];
  
  // Trouver les champs qui diffèrent
  const allFields = new Set([
    ...Object.keys(optimisticState || {}),
    ...Object.keys(serverState || {})
  ]);
  
  for (const field of allFields) {
    if (JSON.stringify(optimisticState[field]) !== JSON.stringify(serverState[field])) {
      differences.push(field);
    }
  }
  
  if (differences.length === 0) {
    return 'Conflit détecté entre les modifications locales et le serveur.';
  }
  
  return `Conflit détecté sur les champs: ${differences.join(', ')}. Le serveur a des valeurs différentes.`;
}

