// Script de test Socket.IO pour 100 connexions simultanées
// Usage: node tests/load_testing/socketio_100_connections.js

const io = require('socket.io-client');

const BASE_URL = process.env.BASE_URL || 'http://localhost:5000';
const TOKEN = process.env.TOKEN || '';
const NUM_CONNECTIONS = parseInt(process.env.NUM_CONNECTIONS || '100');
const DURATION_MS = parseInt(process.env.DURATION_MS || '300000'); // 5 minutes

const sockets = [];
let connected = 0;
let disconnected = 0;
let eventsReceived = 0;
let errors = 0;
let startTime = Date.now();

// Créer connexions
async function createConnections() {
  console.log(`Creating ${NUM_CONNECTIONS} connections to ${BASE_URL}...`);
  
  for (let i = 0; i < NUM_CONNECTIONS; i++) {
    const socket = io(BASE_URL, {
      path: '/socket.io',
      auth: { token: TOKEN },
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 5000,
    });
    
    socket.on('connect', () => {
      connected++;
      if (connected % 10 === 0) {
        console.log(`[${i}] Connected (total: ${connected})`);
      }
      
      // Join room
      socket.emit('join_company');
    });
    
    socket.on('disconnect', (reason) => {
      disconnected++;
      console.log(`[${i}] Disconnected: ${reason} (total disconnected: ${disconnected})`);
    });
    
    socket.on('error', (error) => {
      errors++;
      console.error(`[${i}] Error:`, error);
    });
    
    // Écouter événements
    socket.on('new_reservation', (data) => {
      eventsReceived++;
    });
    
    socket.on('dispatch_progress', (data) => {
      eventsReceived++;
    });
    
    socket.on('booking_updated', (data) => {
      eventsReceived++;
    });
    
    sockets.push(socket);
    
    // Délai entre connexions (éviter storm)
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  console.log(`All ${NUM_CONNECTIONS} connections initiated. Waiting for connections...`);
  await new Promise(resolve => setTimeout(resolve, 5000)); // Attendre 5s pour connexions
  console.log(`Connected: ${connected}/${NUM_CONNECTIONS}`);
}

// Attendre et mesurer
async function runTest() {
  await createConnections();
  
  console.log(`\nTest running for ${DURATION_MS / 1000}s...`);
  console.log(`Monitoring connections...`);
  
  // Afficher stats toutes les 30s
  const statsInterval = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    console.log(`[${elapsed.toFixed(0)}s] Connected: ${connected}, Disconnected: ${disconnected}, Events: ${eventsReceived}, Errors: ${errors}`);
  }, 30000);
  
  await new Promise(resolve => setTimeout(resolve, DURATION_MS));
  clearInterval(statsInterval);
  
  // Statistiques finales
  const totalAttempts = connected + disconnected;
  const disconnectRate = totalAttempts > 0 ? disconnected / totalAttempts : 0;
  const avgEventsPerConnection = connected > 0 ? eventsReceived / connected : 0;
  
  console.log('\n=== Final Results ===');
  console.log(`Total connections attempted: ${totalAttempts}`);
  console.log(`Connected: ${connected}`);
  console.log(`Disconnected: ${disconnected}`);
  console.log(`Disconnect rate: ${(disconnectRate * 100).toFixed(2)}%`);
  console.log(`Events received: ${eventsReceived}`);
  console.log(`Avg events per connection: ${avgEventsPerConnection.toFixed(2)}`);
  console.log(`Errors: ${errors}`);
  console.log(`Test duration: ${(Date.now() - startTime) / 1000}s`);
  
  // Fermer connexions
  console.log('\nClosing connections...');
  sockets.forEach(socket => socket.close());
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Critère de succès
  if (disconnectRate < 0.01 && connected >= NUM_CONNECTIONS * 0.95) {
    console.log('\n✅ SUCCESS: Disconnect rate < 1% and >= 95% connections established');
    process.exit(0);
  } else {
    console.log('\n❌ FAIL: Disconnect rate >= 1% or < 95% connections established');
    process.exit(1);
  }
}

// Gestion erreurs
process.on('unhandledRejection', (error) => {
  console.error('Unhandled rejection:', error);
  process.exit(1);
});

runTest().catch((error) => {
  console.error('Test failed:', error);
  process.exit(1);
});
