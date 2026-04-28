import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import apiClient from '../../utils/apiClient';
import { getActiveUser, setEnvUser } from '../../utils/webAuthSession';

const ResetPassword = () => {
  const { token, userId } = useParams();
  const navigate = useNavigate();

  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isForced, setIsForced] = useState(false);

  useEffect(() => {
    const user = getActiveUser();
    if (user && user.force_password_change) {
      console.log('⚠️ Réinitialisation forcée du mot de passe requise.');
      setIsForced(true);
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');

    // Vérification basique : mots de passe identiques
    if (newPassword !== confirmPassword) {
      setError('Les mots de passe ne correspondent pas.');
      return;
    }

    console.log('📌 Soumission du formulaire de réinitialisation');
    console.log("🔗 Token extrait de l'URL :", token);
    console.log("🔗 userId extrait de l'URL :", userId);

    setIsLoading(true);

    try {
      let response;

      // ---------------------------------------------------------------------
      // 1) Mode "réinitialisation par token" (lien e-mail)
      //    => URL /reset-password/:token
      // ---------------------------------------------------------------------
      if (token) {
        response = await apiClient.post(`/auth/reset-password/${token}`, {
          new_password: newPassword,
          confirm_password: confirmPassword,
        });

        console.log('✅ Réinitialisation (par token) réussie :', response.data);
        setMessage('Votre mot de passe a été mis à jour avec succès !');

        // Redirection standard vers le login
        navigate('/login');

        // ---------------------------------------------------------------------
        // 2) Mode "réinitialisation forcée par userId" (admin)
        //    => URL /force-reset-password/:userId
        // ---------------------------------------------------------------------
      } else if (userId) {
        response = await apiClient.post(`/auth/reset-password/${userId}`, {
          new_password: newPassword,
          confirm_password: confirmPassword,
        });

        console.log('✅ Réinitialisation (par userId) réussie :', response.data);
        setMessage('Mot de passe réinitialisé avec succès !');

        // Mise à jour du localStorage, si on a un user forcé
        const updatedUser = getActiveUser();
        if (updatedUser) {
          updatedUser.force_password_change = false;
          setEnvUser(updatedUser);

          // Redirection vers le dashboard (ou autre)
          setTimeout(() => {
            navigate(`/dashboard/${updatedUser?.role}/${updatedUser?.public_id}`);
          }, 2000);
        } else {
          // Si aucun user dans le localStorage, on redirige éventuellement ailleurs
          setTimeout(() => {
            navigate('/login');
          }, 2000);
        }
      } else {
        // Ni token ni userId ? On prévient l’utilisateur.
        setError("Aucun token ni identifiant utilisateur n'est fourni.");
        return;
      }
    } catch (err) {
      console.error('❌ Erreur lors de la réinitialisation :', err);
      setError(
        err.response?.data?.error ||
          'Une erreur est survenue lors de la réinitialisation du mot de passe.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-sm mx-auto mt-10 p-4 border rounded shadow-lg bg-white">
      <h2 className="text-xl font-bold mb-4 text-center">
        {isForced ? '🔒 Modification obligatoire du mot de passe' : 'Réinitialiser le mot de passe'}
      </h2>
      <form onSubmit={handleSubmit}>
        <input
          type="password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          placeholder="Nouveau mot de passe"
          required
          className="w-full p-2 mb-4 border rounded"
        />
        <input
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          placeholder="Confirmer le mot de passe"
          required
          className="w-full p-2 mb-4 border rounded"
        />
        <button
          type="submit"
          className="w-full bg-green-500 hover:bg-green-600 text-white py-2 rounded transition duration-200"
          disabled={isLoading}
        >
          {isLoading ? 'Réinitialisation...' : 'Confirmer le changement'}
        </button>
      </form>

      {message && <p className="text-green-500 mt-4 text-center">{message}</p>}
      {error && <p className="text-red-500 mt-4 text-center">{error}</p>}
    </div>
  );
};

export default ResetPassword;
