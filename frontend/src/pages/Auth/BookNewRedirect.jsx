import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useAuthToken from '../../hooks/useAuthToken';

const BookNewRedirect = () => {
  const user = useAuthToken();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) return;

    const role = String(user.role || '').toLowerCase();
    const publicId = user.public_id || user.sub || '';

    if (role === 'client' && publicId) {
      navigate(`/reservations/${publicId}`, { replace: true });
      return;
    }

    navigate('/dashboard', { replace: true });
  }, [user, navigate]);

  return <div>Redirection vers votre réservation…</div>;
};

export default BookNewRedirect;
