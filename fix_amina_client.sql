-- Diagnostic: Client 24251 = Amina NDUKWA (Route d'Hermance 63, OPAD)
-- User 78820 a ete ecrase avec Demetrios XOUDIS (saut de personne)
-- La facture EM-2026-01-0076 contient les courses d'Amina (30332, 30333)

-- 1. Restaurer User 78820 = Amina NDUKWA
UPDATE "user" SET
  first_name = 'Amina',
  last_name = 'NDUKWA',
  address = 'Route d''Hermance 63, 1245, Collonge-Bellerive'
WHERE id = 78820;

-- 2. Corriger Client 24251 - domicile
UPDATE client SET
  domicile_address = 'Route d''Hermance 63, 1245, Collonge-Bellerive',
  domicile_zip = '1245',
  domicile_city = 'Collonge-Bellerive'
WHERE id = 24251;
