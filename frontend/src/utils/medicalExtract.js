/**
 * Extrait les éléments médicaux clés d'un texte libre (adresse, email, notes, etc.)
 * Retourne un objet { medical_facility, hospital_service, building, floor, doctor_name }
 */

/** Mots d'établissement / géographie à ne jamais traiter comme « service ». */
const SERVICE_BLOCKLIST =
  /^(h[oô]pitaux?|universitaires?|clinique|ems|centre|maison|foyer|r[ée]sidence|gen[eè]ve|lausanne|zurich|suisse|switzerland)$/i;

/**
 * Nom d'établissement avant la première virgule.
 * Conserve le libellé complet, y compris l'acronyme entre parenthèses
 * (ex. "Hôpitaux Universitaires de Genève (HUG)").
 */
export function extractEstablishmentLabel(text) {
  if (!text || typeof text !== 'string') return '';
  return text.split(',')[0].trim();
}

function looksLikeMedicalFacility(text) {
  if (!text) return false;
  return /(h[oô]pital|h[oô]pitaux|clinique|ems\b|chu?\b|hug\b|centre m[ée]dical|maison m[ée]dicale|polyclinique|ehpad|foyer|r[ée]sidence|imagerie|radiologie)/i.test(
    text
  );
}

export function extractMedicalServiceInfo(text) {
  if (!text) return {};

  const result = {};

  // 1. Extraction Docteur
  const doctorRegex =
    /(dr\.?|docteur|docteure|prof\.?)\s*([A-ZÉÈ][a-zéèêëîïôöûüàâäç\-']{2,}(?:\s+[A-ZÉÈ][a-zéèêëîïôöûüàâäç\-']{2,}){1,2})/i;
  const doctorMatch = text.match(doctorRegex);
  if (doctorMatch) {
    const match = doctorMatch[0];
    const split = match.split(/\s(?=\d)/)[0]; // coupe avant premier chiffre (numéro rue)
    const name = split
      .replace(/\b(M[ée]d\.?|Medecin|Médecin)\b/gi, '')
      .replace(/\s+/g, ' ')
      .trim();
    result.doctor_name = name;
  }

  // 2. Extraction Service — exiger un préfixe métier OU un suffixe de spécialité.
  // Ne plus matcher n'importe quel mot capitalisé (ex. "Hôpitaux").
  const serviceWithPrefix =
    /(?:unité|service|département|secteur|pôle)\s+(?:d['’]\s*)?([A-Za-zÀ-ÿ][\wÀ-ÿ'’\- ]{1,40})/i;
  const specialtyOnly =
    /\b([A-ZÉÈ][a-zéèêëîïôöûüàâäç]+(?:ologie|iatrie|graphie|pathie|oscopie|thérapie|urgences?))\b/;

  const prefixMatch = text.match(serviceWithPrefix);
  if (prefixMatch) {
    const candidate = (prefixMatch[1] || prefixMatch[0] || '').trim();
    if (candidate && !SERVICE_BLOCKLIST.test(candidate.split(/\s+/)[0])) {
      result.hospital_service = prefixMatch[0].trim();
    }
  } else {
    const specialtyMatch = text.match(specialtyOnly);
    if (specialtyMatch) {
      const candidate = specialtyMatch[1].trim();
      if (!SERVICE_BLOCKLIST.test(candidate)) {
        result.hospital_service = candidate;
      }
    }
  }

  // 3. Extraction bâtiment (Bâtiment, Building, Aile, Pavillon, etc.)
  const buildingRegex = /(bâtiment|bât|building|aile|pavillon|tour|bloc)\s+[A-Za-z0-9-]+/i;
  const buildingMatch = text.match(buildingRegex);
  if (buildingMatch) {
    result.building = buildingMatch[0].trim();
  }

  // 4. Extraction étage (étage, floor, level, 1er, 2ème, etc.)
  const floorRegex =
    /([0-9]{1,2}(er|ème|e)?\s?étage|étage\s?[0-9]{1,2}|level\s?[0-9]{1,2}|floor\s?[0-9]{1,2})/i;
  const floorMatch = text.match(floorRegex);
  if (floorMatch) {
    result.floor = floorMatch[0].trim();
  }

  // 5. Établissement : nom complet avant virgule (ex. "Hôpitaux Universitaires de Genève (HUG)").
  const head = text.split(',')[0].trim();
  if (looksLikeMedicalFacility(head) || looksLikeMedicalFacility(text)) {
    const label = extractEstablishmentLabel(text);
    if (label) {
      result.medical_facility = label;
    }
  }

  return result;
}
