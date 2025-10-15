describe('Company Flow - Login to Booking Management', () => {
  beforeEach(() => {
    cy.viewport(1280, 720)
    
    // Mock API responses
    cy.intercept('POST', '/api/auth/login').as('login')
    cy.intercept('GET', '/api/companies/me').as('getCompany')
    cy.intercept('GET', '/api/companies/me/bookings*').as('getBookings')
    cy.intercept('GET', '/api/companies/me/drivers').as('getDrivers')
  })
  
  it('logs in as company and views dashboard', () => {
    cy.visit('/login')
    
    cy.get('input[name="email"]').type('company@test.com')
    cy.get('input[name="password"]').type('password123')
    cy.get('button[type="submit"]').click()
    
    cy.wait('@login').its('response.statusCode').should('eq', 200)
    
    // Redirect vers dashboard
    cy.url().should('include', '/company/dashboard')
    
    cy.wait('@getCompany')
    cy.wait('@getBookings')
    
    // Vérifier affichage nom entreprise
    cy.contains('Test Transport SA')
  })
  
  it('creates booking and assigns driver', () => {
    // Login préalable
    cy.login('company@test.com', 'password123')
    cy.visit('/company/bookings')
    
    cy.wait('@getBookings')
    
    // Cliquer "Nouvelle réservation"
    cy.get('[data-testid="create-booking-btn"]').click()
    
    // Remplir formulaire
    cy.get('input[name="customer_name"]').type('Patient Test')
    cy.get('input[name="pickup_location"]').type('Genève, Rue du Rhône 1')
    cy.get('input[name="dropoff_location"]').type('Lausanne, CHUV')
    cy.get('input[name="scheduled_time"]').type('2025-10-20T14:00')
    cy.get('input[name="amount"]').type('45.50')
    
    cy.intercept('POST', '/api/bookings/clients/*/bookings').as('createBooking')
    cy.get('button[type="submit"]').click()
    
    cy.wait('@createBooking').its('response.statusCode').should('eq', 201)
    
    // Vérifier notification succès
    cy.contains(/réservation créée/i)
    
    // Assigner chauffeur
    cy.wait('@getBookings')
    cy.get('[data-testid="booking-row"]').first().click()
    
    cy.get('[data-testid="assign-driver-select"]').select('Driver A')
    cy.intercept('POST', '/api/bookings/*/assign').as('assignDriver')
    cy.get('[data-testid="confirm-assign-btn"]').click()
    
    cy.wait('@assignDriver').its('response.statusCode').should('eq', 200)
    cy.contains(/chauffeur assigné/i)
  })
  
  it('triggers dispatch and views assignments', () => {
    cy.login('company@test.com', 'password123')
    cy.visit('/company/dispatch')
    
    // Sélectionner date
    cy.get('input[name="dispatch_date"]').type('2025-10-20')
    
    cy.intercept('POST', '/api/companies/me/dispatch/run').as('runDispatch')
    cy.get('[data-testid="run-dispatch-btn"]').click()
    
    // Attendre résultat (peut prendre 5-10s)
    cy.wait('@runDispatch', { timeout: 15000 })
      .its('response.statusCode')
      .should('eq', 200)
    
    // Vérifier affichage assignments
    cy.get('[data-testid="assignments-list"]').should('be.visible')
    cy.get('[data-testid="assignment-card"]').should('have.length.greaterThan', 0)
  })
})

