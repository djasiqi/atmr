/**
 * Script de test pour valider le parsing d'établissements
 * Tests pour la Priority 1 : residence_facility
 */

import { parseAddressWithEstablishment } from '../addressParser.js';

// Cas de test
const testCases = [
  {
    name: 'Clinique avec adresse complète',
    input: {
      label: 'Clinique La Source, Avenue Vinet 30, 1004, Lausanne',
      item: {}
    },
    expected: {
      establishment: 'Clinique La Source',
      street: 'Avenue Vinet',
      streetNumber: '30',
      postcode: '1004',
      city: 'Lausanne'
    }
  },
  {
    name: 'EMS avec adresse complète',
    input: {
      label: 'EMS Maison de Vessy, Chemin de Vessy 10, 1234, Vessy',
      item: {}
    },
    expected: {
      establishment: 'EMS Maison de Vessy',
      street: 'Chemin de Vessy',
      streetNumber: '10',
      postcode: '1234',
      city: 'Vessy'
    }
  },
  {
    name: 'Hôpital universitaire',
    input: {
      label: 'Hôpital Universitaire de Genève, Rue Gabrielle-Perret-Gentil 4, 1205, Genève',
      item: {}
    },
    expected: {
      establishment: 'Hôpital Universitaire de Genève',
      street: 'Rue Gabrielle-Perret-Gentil',
      streetNumber: '4',
      postcode: '1205',
      city: 'Genève'
    }
  },
  {
    name: 'Adresse sans établissement (domicile classique)',
    input: {
      label: 'Avenue Ernest-Pictet 9, 1203, Genève',
      item: {}
    },
    expected: {
      establishment: '',
      street: 'Avenue Ernest-Pictet',
      streetNumber: '9',
      postcode: '1203',
      city: 'Genève'
    }
  },
  {
    name: "Nom d'autocomplete égal à une rue (ne pas remplir residence_facility)",
    input: {
      label: 'Rue de Lausanne 3, 1201, Genève',
      item: {
        name: 'Rue de Lausanne 3',
        street: 'Rue de Lausanne',
        housenumber: '3',
        postcode: '1201',
        city: 'Genève'
      }
    },
    expected: {
      establishment: '',
      street: 'Rue de Lausanne',
      streetNumber: '3',
      postcode: '1201',
      city: 'Genève'
    }
  },
  {
    name: 'Résidence avec foyer',
    input: {
      label: 'Foyer Clair Bois, Route de Chancy 59, 1213, Petit-Lancy',
      item: {}
    },
    expected: {
      establishment: 'Foyer Clair Bois',
      street: 'Route de Chancy',
      streetNumber: '59',
      postcode: '1213',
      city: 'Petit-Lancy'
    }
  },
  {
    name: 'Item avec composants déjà parsés (autocomplete)',
    input: {
      label: 'Clinique de Genolier',
      item: {
        street: 'Route du Muids',
        housenumber: '3',
        postcode: '1272',
        city: 'Genolier'
      }
    },
    expected: {
      establishment: 'Clinique de Genolier',
      street: 'Route du Muids',
      streetNumber: '3',
      postcode: '1272',
      city: 'Genolier'
    }
  }
];

// Fonction de test
function runTests() {
  console.log('🧪 Test de parsing d\'établissement (residence_facility)\n');
  console.log('='.repeat(80));
  
  let passed = 0;
  let failed = 0;
  
  testCases.forEach((testCase, index) => {
    console.log(`\n📝 Test ${index + 1}: ${testCase.name}`);
    console.log(`Input: "${testCase.input.label}"`);
    
    const result = parseAddressWithEstablishment(testCase.input.label, testCase.input.item);
    
    const matches = {
      establishment: result.establishment === testCase.expected.establishment,
      street: result.street === testCase.expected.street,
      streetNumber: result.streetNumber === testCase.expected.streetNumber,
      postcode: result.postcode === testCase.expected.postcode,
      city: result.city === testCase.expected.city
    };
    
    const allMatch = Object.values(matches).every(m => m);
    
    if (allMatch) {
      console.log('✅ PASS');
      passed++;
    } else {
      console.log('❌ FAIL');
      failed++;
      console.log('\nAttendu:');
      console.log(JSON.stringify(testCase.expected, null, 2));
      console.log('\nReçu:');
      console.log(JSON.stringify(result, null, 2));
      console.log('\nDifférences:');
      Object.entries(matches).forEach(([key, match]) => {
        if (!match) {
          console.log(`  ❌ ${key}: attendu "${testCase.expected[key]}", reçu "${result[key]}"`);
        }
      });
    }
  });
  
  console.log('\n' + '='.repeat(80));
  console.log(`\n📊 Résultats: ${passed} réussis, ${failed} échoués sur ${testCases.length} tests`);
  
  if (failed === 0) {
    console.log('\n🎉 Tous les tests sont passés !');
  } else {
    console.log('\n⚠️  Certains tests ont échoué');
  }
}

// Exécuter les tests
runTests();
