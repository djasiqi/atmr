-- Bookings par client pour confirmer
SELECT client_id, COUNT(*) as nb, MIN(customer_name) as sample_name
FROM booking WHERE client_id IN (24212, 24251) GROUP BY client_id;
