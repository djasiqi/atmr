/**
 * Extrait les éléments médicaux clés d'un texte libre (adresse, email, notes, etc.)
 * Retourne un objet { medical_facility, hospital_service, building, floor, doctor_name }
 */
export function extractMedicalServiceInfo(text) {
  if (!text) return {};

  const result = {};

  // 1. Extraction Docteur
  const doctorRegex =
    /(dr\.?|docteur|docteure|prof\.?)\s*([A-ZÉÈ][a-zéèêëîïôöûüàâäç\-']{2,}(?:\s+[A-ZÉÈ][a-zéèêëîïôöûüàâäç\-']{2,}){1,2})/i;
  const doctorMatch = text.match(doctorRegex);
  if (doctorMatch) {
    let match = doctorMatch[0];
    let split = match.split(/\s(?=\d)/)[0]; // coupe avant premier chiffre (numéro rue)
    let name = split
      .replace(/\b(M[ée]d\.?|Medecin|Médecin)\b/gi, '')
      .replace(/\s+/g, ' ')
      .trim();
    result.doctor_name = name;
  }

  // 2. Extraction Service (Oncologie, Radiologie, etc.)
  const serviceRegex =
    /(unité|service|département|secteur|pôle)?\s?(d['’])?([A-Z][a-zéèêëîïôöûüàâäç]+(ologie|iatrie|graphie|pathie|ie)?)/i;
  const serviceMatch = text.match(serviceRegex);
  if (serviceMatch) {
    result.hospital_service = serviceMatch[0].trim();
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

  // 5. Extraction établissement (recherche mot-clé hôpital, clinique, EMS...)
  const facilityRegex = /(hôpital|hopital|clinique|ems|ch|hug|centre médical|maison médical[e]?)/i;
  const facilityMatch = text.match(facilityRegex);
  if (facilityMatch) {
    // Tente de récupérer la phrase complète jusqu'à la virgule suivante ou fin du mot
    const after = text.slice(facilityMatch.index).split(',')[0];
    result.medical_facility = after.trim();
  }

  return result;
}
