-- 1. Restaurer User 78780 = Demetrios Xoudis (client 24212)
UPDATE "user" SET
  first_name = 'Demetrios',
  last_name = 'Xoudis',
  address = 'Rue Le-Corbusier 29, 1208, Genève',
  birth_date = '1931-05-23'
WHERE id = 78780;

-- 2. Restaurer Client 24212 = Demetrios (domicile)
UPDATE client SET
  domicile_address = 'Rue Le-Corbusier 29, 1208, Genève',
  domicile_zip = '1208',
  domicile_city = 'Genève'
WHERE id = 24212;

-- 3. Enlever le telephone d'Amina (client 24251)
UPDATE client SET contact_phone = NULL WHERE id = 24251;
UPDATE "user" SET phone = NULL WHERE id = 78820;
