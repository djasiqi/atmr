SELECT b.id, b.billed_to_type, b.billed_to_company_id, c.name as billed_to_name FROM booking b LEFT JOIN company c ON b.billed_to_company_id = c.id WHERE b.id = 30889;
