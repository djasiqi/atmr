// src/pages/client/Account/AccountUser.jsx
import React, { useEffect, useState, useRef } from 'react';
import apiClient from '../../../utils/apiClient';
import { useParams, useNavigate } from 'react-router-dom';
// import { Autocomplete } from "@react-google-maps/api"; // ❌ retiré
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import './AccountUser.css';

import avatarMale from '../../../assets/images/avatar-male.png';
import avatarFemale from '../../../assets/images/avatar-female.png';
import defaultAvatar from '../../../assets/images/default-avatar.png';

const AccountUser = () => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const [, setProfile] = useState({});
  const [updatedProfile, setUpdatedProfile] = useState({});
  const [paymentMethod, setPaymentMethod] = useState('none');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [profilePic, setProfilePic] = useState('/default-avatar.png');

  // const autocompleteRef = useRef(null); // ❌ retiré (Google)
  const fileInputRef = useRef(null);

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (!token) {
      navigate('/login');
      return;
    }

    apiClient
      .get(`/clients/${public_id}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      .then((response) => {
        setProfile(response.data);
        setUpdatedProfile(response.data);
        updateProfilePic(response.data.gender, response.data.profile_image);
      })
      .catch(() => {
        setError('Impossible de charger le compte utilisateur.');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [public_id, navigate]);

  const updateProfilePic = (gender, uploadedPic) => {
    if (uploadedPic) {
      setProfilePic(uploadedPic);
    } else {
      switch (gender) {
        case 'Homme':
          setProfilePic(avatarMale);
          break;
        case 'Femme':
          setProfilePic(avatarFemale);
          break;
        default:
          setProfilePic(defaultAvatar);
          break;
      }
    }
  };

  // ❌ onPlaceSelected / autocomplete supprimés

  const handleUpdateProfile = () => {
    const token = localStorage.getItem('authToken');
    apiClient
      .put(`/clients/${public_id}`, updatedProfile, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      })
      .then((response) => {
        alert('Profil mis à jour avec succès !');
        setProfile(response.data.client);
        setUpdatedProfile(response.data.client);
        updateProfilePic(response.data.client.gender, response.data.client.profile_image);
        localStorage.setItem('user', JSON.stringify(response.data.client));
      })
      .catch(() => {
        setError('Impossible de mettre à jour le compte.');
      });
  };

  const handleProfilePicUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setProfilePic(reader.result);
        setUpdatedProfile({ ...updatedProfile, profile_image: reader.result });
      };
      reader.readAsDataURL(file);
    }
  };

  if (loading) return <p>Chargement...</p>;
  if (error) return <p className="error">{error}</p>;

  return (
    <div className="account-container">
      <HeaderDashboard />
      <main className="account-content">
        <h1 className="title">Gestion du Compte</h1>

        <section className="profile-card">
          <div className="profile-header">
            <img src={profilePic} alt="Profil utilisateur" className="profile-pic" />
            <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              onChange={handleProfilePicUpload}
              style={{ display: 'none' }}
            />
            <button className="edit-button" onClick={() => fileInputRef.current?.click()}>
              Modifier la photo
            </button>
          </div>

          <div className="profile-info">
            <label>Prénom:</label>
            <input
              type="text"
              value={updatedProfile.first_name || ''}
              onChange={(e) =>
                setUpdatedProfile({
                  ...updatedProfile,
                  first_name: e.target.value,
                })
              }
            />

            <label>Nom:</label>
            <input
              type="text"
              value={updatedProfile.last_name || ''}
              onChange={(e) =>
                setUpdatedProfile({
                  ...updatedProfile,
                  last_name: e.target.value,
                })
              }
            />

            <label>Email:</label>
            <input type="email" value={updatedProfile.email || ''} disabled />

            <label>Téléphone:</label>
            <input
              type="text"
              value={updatedProfile.phone || ''}
              onChange={(e) => setUpdatedProfile({ ...updatedProfile, phone: e.target.value })}
            />

            <label>Adresse complète:</label>
            {/* ✅ Champ d’adresse simple (contrôlé) */}
            <input
              type="text"
              value={updatedProfile.address || ''}
              placeholder="Saisissez votre adresse…"
              onChange={(e) =>
                setUpdatedProfile({
                  ...updatedProfile,
                  address: e.target.value,
                })
              }
            />

            <label>Date de Naissance:</label>
            <input
              type="date"
              value={updatedProfile.birth_date || ''}
              onChange={(e) =>
                setUpdatedProfile({
                  ...updatedProfile,
                  birth_date: e.target.value,
                })
              }
            />

            <label>Genre:</label>
            <select
              value={updatedProfile.gender || ''}
              onChange={(e) => {
                setUpdatedProfile({
                  ...updatedProfile,
                  gender: e.target.value,
                });
                updateProfilePic(e.target.value, updatedProfile.profile_image);
              }}
            >
              <option value="">Sélectionner...</option>
              <option value="Homme">Homme</option>
              <option value="Femme">Femme</option>
              <option value="Autre">Autre</option>
            </select>
          </div>

          <button className="save-button" onClick={handleUpdateProfile}>
            Sauvegarder
          </button>
        </section>

        <section className="payment-section">
          <h2>Moyens de Paiement</h2>
          <select onChange={(e) => setPaymentMethod(e.target.value)}>
            <option value="none">Sélectionner un moyen de paiement</option>
            <option value="card">Carte Bancaire</option>
            <option value="twint">Twint</option>
            <option value="invoice">Paiement par Facture</option>
          </select>
          {paymentMethod !== 'none' && (
            <button className="add-payment">Ajouter ce moyen de paiement</button>
          )}
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default AccountUser;
