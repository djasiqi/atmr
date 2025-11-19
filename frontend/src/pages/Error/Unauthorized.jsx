// src/pages/Dashboard/Unauthorized.jsx
import React from 'react';
import { Link } from 'react-router-dom';

const Unauthorized = () => {
  return (
    <div style={{ padding: '2rem', textAlign: 'center' }}>
      <h1>Accès non autorisé</h1>
      <p>Vous n'êtes pas autorisé à accéder à cette page.</p>
      <Link to="/">Retour à l'accueil</Link>
    </div>
  );
};

export default Unauthorized;
