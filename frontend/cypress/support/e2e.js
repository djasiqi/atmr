// ***********************************************************
// Support file Cypress - Commandes custom
// ***********************************************************

import './commands'

// ✅ Gestion exceptions non catchées
Cypress.on('uncaught:exception', (err, runnable) => {
  // Ignore certaines erreurs React/Redux non critiques
  if (err.message.includes('ResizeObserver')) {
    return false
  }
  return true
})

