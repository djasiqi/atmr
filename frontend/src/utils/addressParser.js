/**
 * Utilitaire pour parser les adresses et détecter les établissements
 * (cliniques, hôpitaux, EMS, foyers, etc.)
 */

// Mots-clés pour détecter les établissements
const ESTABLISHMENT_KEYWORDS = [
  'clinique',
  'hôpital',
  'hopital',
  'hospital',
  'ems',
  'foyer',
  'centre',
  'maison',
  'résidence',
  'residence',
  'institut',
  'institution',
  'établissement',
  'etablissement',
  'cabinet',
  'dispensaire',
  'polyclinique',
  'sanatorium',
  'maison de santé',
  'maison de retraite',
  'ehpad',
  'usld',
  'chru',
  'chu',
  'ch',
  'hug',
  'hôpital universitaire',
  'hopital universitaire',
];

/**
 * Détecte si une chaîne contient un mot-clé d'établissement (mot entier uniquement).
 * Ex. "ch" ne matche pas "Chemin" (évite rue "Chemin des Courbes" prise pour établissement).
 */
function isEstablishment(text) {
  if (!text || typeof text !== 'string') return false;
  const lowerText = text.toLowerCase();
  return ESTABLISHMENT_KEYWORDS.some((keyword) => {
    const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp('\\b' + escaped + '\\b', 'i');
    return re.test(lowerText);
  });
}

/**
 * Parse une adresse et extrait l'établissement, la rue, le code postal et la ville
 * 
 * @param {string} label - Label complet de l'adresse (ex: "Clinique X, Rue Y, 1200, Ville")
 * @param {object} item - Item de l'autocomplete avec postcode, city, street, etc.
 * @returns {object} - { establishment, street, streetNumber, postcode, city }
 */
export function parseAddressWithEstablishment(label, item = {}) {
  // Si on a déjà les composants dans item, les utiliser en priorité
  let street = item.street || '';
  let streetNumber = item.street_number || item.housenumber || '';
  let postcode = item.postcode || '';
  let city = item.city || item.locality || '';
  let establishment = '';

  // Si on n'a pas les composants, parser depuis le label
  if (!street && !postcode && !city && label) {
    // Séparer par virgules
    const parts = label.split(',').map((p) => p.trim()).filter((p) => p.length > 0);

    if (parts.length >= 3) {
      // Format possible : "Établissement, Rue, CP, Ville" ou "Établissement, Rue Numéro, CP, Ville"
      const firstPart = parts[0];
      
      // Vérifier si la première partie est un établissement
      if (isEstablishment(firstPart)) {
        establishment = firstPart;
        
        // La deuxième partie est la rue (avec ou sans numéro)
        const streetPart = parts[1];
        const streetMatch = streetPart.match(/^(.+?)\s+(\d+[a-z]*)$/);
        if (streetMatch) {
          street = streetMatch[1].trim();
          streetNumber = streetMatch[2].trim();
        } else {
          street = streetPart;
        }
        
        // La troisième partie peut être le code postal ou la ville
        // Si c'est 4 chiffres, c'est le code postal
        const thirdPart = parts[2];
        if (/^\d{4}$/.test(thirdPart)) {
          postcode = thirdPart;
          // La quatrième partie est la ville
          if (parts.length >= 4) {
            city = parts[3].replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
          }
        } else {
          // La troisième partie est la ville (code postal manquant ou dans la deuxième partie)
          city = thirdPart.replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
          // Essayer d'extraire le code postal de la deuxième partie si possible
          const zipMatch = streetPart.match(/\b(\d{4})\b/);
          if (zipMatch) {
            postcode = zipMatch[1];
            // Retirer le code postal de la rue
            street = street.replace(/\b\d{4}\b/, '').trim();
          }
        }
      } else {
        // Pas d'établissement : format classique "Rue Numéro, CP, Ville"
        const streetPart = parts[0];
        const streetMatch = streetPart.match(/^(.+?)\s+(\d+[a-z]*)$/);
        if (streetMatch) {
          street = streetMatch[1].trim();
          streetNumber = streetMatch[2].trim();
        } else {
          street = streetPart;
        }
        
        // Deuxième partie : code postal ou ville
        const secondPart = parts[1];
        if (/^\d{4}$/.test(secondPart)) {
          postcode = secondPart;
          if (parts.length >= 3) {
            city = parts[2].replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
          }
        } else {
          // Format "Rue, CP Ville"
          const zipCityMatch = secondPart.match(/^(\d{4})\s+(.+?)(?:\s*,\s*(?:Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia))?$/i);
          if (zipCityMatch) {
            postcode = zipCityMatch[1];
            city = zipCityMatch[2].trim();
          } else {
            city = secondPart.replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
          }
        }
      }
    } else if (parts.length === 2) {
      // Format "Rue Numéro, CP Ville" ou "Établissement, Rue"
      const firstPart = parts[0];
      const secondPart = parts[1];
      
      if (isEstablishment(firstPart)) {
        establishment = firstPart;
        // La deuxième partie est la rue
        const streetMatch = secondPart.match(/^(.+?)\s+(\d+[a-z]*)$/);
        if (streetMatch) {
          street = streetMatch[1].trim();
          streetNumber = streetMatch[2].trim();
        } else {
          street = secondPart;
        }
        // Code postal et ville manquants dans ce format
      } else {
        // Format classique "Rue Numéro, CP Ville"
        const streetMatch = firstPart.match(/^(.+?)\s+(\d+[a-z]*)$/);
        if (streetMatch) {
          street = streetMatch[1].trim();
          streetNumber = streetMatch[2].trim();
        } else {
          street = firstPart;
        }
        
        const zipCityMatch = secondPart.match(/^(\d{4})\s+(.+?)(?:\s*,\s*(?:Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia))?$/i);
        if (zipCityMatch) {
          postcode = zipCityMatch[1];
          city = zipCityMatch[2].trim();
        } else {
          city = secondPart.replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
        }
      }
    }
  }

  // Si on a les composants de item mais pas d'établissement, vérifier le label
  if (!establishment && label && isEstablishment(label)) {
    // Essayer d'extraire l'établissement du label
    const parts = label.split(',').map((p) => p.trim());
    if (parts.length > 0 && isEstablishment(parts[0])) {
      establishment = parts[0];
    }
  }

  return {
    establishment: establishment.trim(),
    street: street.trim(),
    streetNumber: streetNumber.trim(),
    postcode: postcode.trim(),
    city: city.trim(),
  };
}

