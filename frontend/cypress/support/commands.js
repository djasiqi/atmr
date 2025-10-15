// ***********************************************
// Custom commands Cypress
// ***********************************************

/**
 * Login command - simplifie authentification dans tests
 */
Cypress.Commands.add('login', (email, password) => {
  cy.request({
    method: 'POST',
    url: `${Cypress.env('apiUrl')}/auth/login`,
    body: { email, password }
  }).then((response) => {
    expect(response.status).to.eq(200)
    const { token, refresh_token, user } = response.body
    
    // Store tokens
    window.localStorage.setItem('authToken', token)
    window.localStorage.setItem('refreshToken', refresh_token)
    window.localStorage.setItem('user', JSON.stringify(user))
  })
})

/**
 * Logout command
 */
Cypress.Commands.add('logout', () => {
  window.localStorage.removeItem('authToken')
  window.localStorage.removeItem('refreshToken')
  window.localStorage.removeItem('user')
  cy.visit('/login')
})

/**
 * Seed test data
 */
Cypress.Commands.add('seedTestData', () => {
  // Call backend seeding endpoint (à créer si nécessaire)
  cy.request({
    method: 'POST',
    url: `${Cypress.env('apiUrl')}/test/seed`,
    failOnStatusCode: false
  })
})

