// C:\Users\jasiq\atmr\frontend\src\pages\Users\AdminUsers.jsx
import React, { useEffect, useState } from 'react';
import apiClient from '../../../utils/apiClient';
import {
  fetchUsers,
  deleteUser,
  resetUserPassword,
  updateUserRole, // ✅ Utilisation de la version du service
  fetchCompanies,
  fetchInstitutions,
} from '../../../services/adminService';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import AdminSidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import styles from './AdminUsers.module.css';

const AdminUsers = () => {
  const [users, setUsers] = useState([]);
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [loading, setLoading] = useState(true);
  const [companyOptions, setCompanyOptions] = useState([]);
  const [showCompanyDropdown, setShowCompanyDropdown] = useState(false);
  const [pendingDriverUserId, setPendingDriverUserId] = useState(null);
  // ✅ États pour les institutions
  const [institutionOptions, setInstitutionOptions] = useState([]);
  const [showInstitutionDropdown, setShowInstitutionDropdown] = useState(false);
  const [pendingInstitutionUserId, setPendingInstitutionUserId] = useState(null);
  const [selectedInstitutionId, setSelectedInstitutionId] = useState(null);
  const [selectedInstitutionRole, setSelectedInstitutionRole] = useState('institution_admin');
  const norm = (v) => String(v ?? '').toLowerCase();

  useEffect(() => {
    const loadUsers = async () => {
      setLoading(true);
      try {
        const data = await fetchUsers();
        console.log('📌 Utilisateurs chargés dans AdminUsers :', data);
        setUsers(data || []);
      } catch (error) {
        console.error('❌ Erreur chargement utilisateurs :', error);
      } finally {
        setLoading(false);
        console.log('🔄 Chargement terminé'); // Vérification
      }
    };

    loadUsers();
  }, []);

  useEffect(() => {
    const loadCompanies = async () => {
      console.log('📡 Tentative de chargement des entreprises...');
      try {
        const companies = await fetchCompanies();
        console.log('✅ Entreprises chargées :', companies);
        // on ajoute un flag selected utilisable par le modal
        setCompanyOptions((companies || []).map((c) => ({ ...c, selected: false })));
      } catch (error) {
        console.error('⚠️ Erreur chargement entreprises :', error);
      }
    };
    loadCompanies();
  }, []);

  // ✅ Chargement des institutions au démarrage
  useEffect(() => {
    const loadInstitutions = async () => {
      console.log('📡 Tentative de chargement des institutions...');
      try {
        const institutions = await fetchInstitutions();
        console.log('✅ Institutions chargées :', institutions);
        setInstitutionOptions(institutions || []);
      } catch (error) {
        console.error('⚠️ Erreur chargement institutions :', error);
      }
    };
    loadInstitutions();
  }, []);

  const updateUserRoleHandler = async (userId, newRole) => {
    if (!userId || !newRole) {
      alert("⚠️ Erreur : L'utilisateur ou le rôle est invalide.");
      return;
    }

    // Vérifier si on assigne le rôle "driver"
    if (newRole.toLowerCase() === 'driver') {
      // Si c'est un chauffeur, on affiche la liste des entreprises dans un modal
      if (!companyOptions.length) {
        alert('❌ Aucune entreprise disponible !');
        return;
      }

      setPendingDriverUserId(userId);
      setShowCompanyDropdown(true);
    } else if (newRole.toLowerCase() === 'institution') {
      // ✅ Si c'est une institution, créer automatiquement l'institution avec le nom de l'utilisateur
      // Le backend va créer l'institution automatiquement si aucune institution_id n'est fournie
      try {
        await updateUserRole(userId, { 
          role: 'institution',
          institution_role: 'institution_admin'  // Admin par défaut
        });
        alert(`✅ Rôle Institution attribué avec succès ! L'institution a été créée automatiquement.`);
        loadUsers();
      } catch (error) {
        console.error('❌ Erreur attribution rôle institution :', error);
        alert('⚠️ Impossible d\'attribuer le rôle institution.');
      }
      return;
    } else {
      // Pour les autres rôles, mise à jour directe
      try {
        await updateUserRole(userId, { role: newRole });
        alert(`✅ Rôle mis à jour avec succès : ${newRole}`);
        loadUsers();
      } catch (error) {
        console.error('❌ Erreur mise à jour rôle :', error);
        alert('⚠️ Impossible de mettre à jour le rôle.');
      }
    }
  };

  const loadUsers = async () => {
    setLoading(true);
    try {
      const data = await fetchUsers();
      console.log('📌 Utilisateurs chargés dans AdminUsers :', data);
      setUsers(data || []);
    } catch (error) {
      console.error('❌ Erreur chargement utilisateurs :', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (userId) => {
    if (!window.confirm('❌ Confirmer la suppression de cet utilisateur ?')) {
      return;
    }
    try {
      await deleteUser(userId);
      loadUsers();
      alert('✅ Utilisateur supprimé avec succès !');
    } catch (error) {
      console.error('❌ Erreur suppression utilisateur :', error);
      alert("⚠️ Impossible de supprimer l'utilisateur.");
    }
  };

  const handleResetPassword = async (userId) => {
    if (!userId) {
      console.error('❌ Erreur : userId est undefined !');
      alert('⚠️ Impossible de réinitialiser le mot de passe : ID utilisateur introuvable.');
      return;
    }

    console.log(`🔄 Tentative de réinitialisation pour l'ID utilisateur : ${userId}`);

    const confirmation = window.confirm(
      'Voulez-vous vraiment réinitialiser le mot de passe de cet utilisateur ?'
    );

    if (!confirmation) return;

    try {
      const response = await resetUserPassword(userId);

      if (response?.new_password) {
        alert(`✅ Mot de passe réinitialisé avec succès : ${response.new_password}`);
        console.log('✅ Nouveau mot de passe généré :', response.new_password);
      } else {
        console.warn('⚠️ La réponse API ne contient pas de mot de passe.');
        alert('⚠️ Échec de la réinitialisation : aucun mot de passe généré.');
      }
    } catch (error) {
      console.error(
        '❌ Erreur lors de la réinitialisation du mot de passe :',
        error.response?.data || error.message
      );
      alert('❌ Une erreur est survenue lors de la réinitialisation.');
    }
  };

  const filteredUsers = users
    .filter((user) => {
      const matchesSearch =
        norm(user.username).includes(norm(search)) ||
        norm(user.email).includes(norm(search));
      const matchesRole = roleFilter ? norm(user.role) === norm(roleFilter) : true;
      return matchesSearch && matchesRole;
    })
    .sort((a, b) => {
      if (sortBy === 'created_at') {
        return new Date(b.created_at) - new Date(a.created_at);
      } else if (sortBy === 'username') {
        return norm(a.username).localeCompare(norm(b.username));
      } else if (sortBy === 'role') {
        return norm(a.role).localeCompare(norm(b.role));
      }
      return 0;
    });

  return (
    <div className={styles.adminContainer}>
      {/* ✅ Intégration du HeaderDashboard */}
      <HeaderDashboard />

      <div className={styles.dashboard}>
        {/* ✅ Intégration de la Sidebar */}
        <AdminSidebar />

        <main className={styles.content}>
          <h1>👥 Gestion des utilisateurs</h1>

          {/* 🔎 Barre de recherche */}
          <input
            type="text"
            placeholder="Rechercher par nom ou email..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className={styles.searchInput}
          />

          {/* 🎭 Filtrer par rôle */}
          <div className={styles.filters}>
            <select
              value={roleFilter}
              onChange={(e) => setRoleFilter(e.target.value)}
              className={styles.roleFilter}
            >
              <option value="">🎭 Tous les rôles</option>
              <option value="admin">🛠️ Admin</option>
              <option value="client">👤 Client</option>
              <option value="driver">🚖 Chauffeur</option>
              <option value="company">🏢 Entreprise</option>
              <option value="institution">🏥 Institution</option>
            </select>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className={styles.roleFilter}
            >
              <option value="created_at">📅 Trier par Date d'inscription</option>
              <option value="username">🔠 Trier par Nom</option>
              <option value="role">🎭 Trier par Rôle</option>
            </select>
          </div>

          {/* 📋 Liste des utilisateurs */}
          <div className={styles.tableContainer}>
            <table className={styles.userTable}>
              <thead>
                <tr>
                  <th>👤 Nom</th>
                  <th>📧 Email</th>
                  <th>🎭 Rôle</th>
                  <th>📅 Date d'inscription</th>
                  <th>⚙️ Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan="5">⏳ Chargement...</td>
                  </tr>
                ) : filteredUsers.length > 0 ? (
                  filteredUsers.map((user) => {
                    console.log("👤 Affichage de l'utilisateur :", user);
                    const userRole = norm(user.role); // <-- normalisation pour le select
                    return (
                      <tr key={user.id}>
                        <td>{user.username}</td>
                        <td>{user.email}</td>
                        <td>
                          <select
                            value={userRole}
                            onChange={(e) => updateUserRoleHandler(user.id, e.target.value)}
                          >
                            <option value="client">👤 Client</option>
                            <option value="company">🏢 Entreprise</option>
                            <option value="driver">🚖 Chauffeur</option>
                            <option value="institution">🏥 Institution</option>
                            <option value="admin">🛠️ Admin</option>
                          </select>
                        </td>

                        <td>
                          {user.created_at
                            ? new Date(user.created_at).toLocaleString('fr-CH')
                            : '📅 Inconnu'}{' '}
                        </td>
                        <td>
                          <button
                            onClick={() => handleResetPassword(user.id)}
                            className={styles.resetButton}
                          >
                            🔑 Réinitialiser
                          </button>
                          <button
                            onClick={() => handleDelete(user.id)}
                            className={styles.deleteButton}
                          >
                            ❌ Supprimer
                          </button>
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan="5">Aucun utilisateur trouvé</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </main>
      </div>
      {showCompanyDropdown && (
        <div className={styles.modal}>
          <div className={styles.modalContent}>
            <h3>Assigner une entreprise au chauffeur</h3>
            <select
              onChange={(e) =>
                setCompanyOptions((prev) =>
                  prev.map((c) => ({
                    ...c,
                    selected: c.id === parseInt(e.target.value, 10),
                  }))
                )
              }
            >
              <option value="">Sélectionnez une entreprise</option>
              {companyOptions.map((company) => (
                <option key={company.id} value={company.id}>
                  {company.name}
                </option>
              ))}
            </select>
            <button
              onClick={async () => {
                // Récupérer la valeur sélectionnée
                const selectedCompany = companyOptions.find((c) => c.selected);
                if (!selectedCompany) {
                  alert('Veuillez sélectionner une entreprise.');
                  return;
                }
                try {
                  const updateData = {
                    role: 'driver',
                    company_id: selectedCompany.id,
                  };
                  const response = await apiClient.put(
                    `/admin/users/${pendingDriverUserId}/role`,
                    updateData
                    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
                  );
                  if (response.status === 200) {
                    alert(`✅ Rôle mis à jour avec succès : driver`);
                    loadUsers();
                    setShowCompanyDropdown(false);
                    setPendingDriverUserId(null);
                  }
                } catch (error) {
                  console.error(
                    '❌ Erreur lors de la mise à jour du rôle :',
                    error.response?.data || error.message
                  );
                  alert('⚠️ Impossible de mettre à jour le rôle.');
                }
              }}
            >
              Valider
            </button>
            <button
              onClick={() => {
                setShowCompanyDropdown(false);
                setPendingDriverUserId(null);
              }}
            >
              Annuler
            </button>
          </div>
        </div>
      )}

      {/* ✅ Modal pour sélection d'institution */}
      {showInstitutionDropdown && (
        <div className={styles.modal}>
          <div className={styles.modalContent}>
            <h3>🏥 Assigner une institution</h3>
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>Institution :</label>
              <select
                value={selectedInstitutionId || ''}
                onChange={(e) => setSelectedInstitutionId(parseInt(e.target.value, 10) || null)}
                style={{ width: '100%', padding: '8px' }}
              >
                <option value="">Sélectionnez une institution</option>
                {institutionOptions.map((inst) => (
                  <option key={inst.id} value={inst.id}>
                    {inst.name} ({inst.type || 'clinique'})
                  </option>
                ))}
              </select>
            </div>
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>Rôle dans l'institution :</label>
              <select
                value={selectedInstitutionRole}
                onChange={(e) => setSelectedInstitutionRole(e.target.value)}
                style={{ width: '100%', padding: '8px' }}
              >
                <option value="institution_admin">🛠️ Admin institution</option>
                <option value="institution_requester">📝 Demandeur</option>
                <option value="institution_reader">👁️ Lecteur</option>
                <option value="institution_billing">💰 Facturation</option>
              </select>
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={async () => {
                  if (!selectedInstitutionId) {
                    alert('Veuillez sélectionner une institution.');
                    return;
                  }
                  try {
                    const updateData = {
                      role: 'institution',
                      institution_id: selectedInstitutionId,
                      institution_role: selectedInstitutionRole,
                    };
                    const response = await apiClient.put(
                      `/admin/users/${pendingInstitutionUserId}/role`,
                      updateData
                    );
                    if (response.status === 200) {
                      alert(`✅ Rôle mis à jour avec succès : institution`);
                      loadUsers();
                      setShowInstitutionDropdown(false);
                      setPendingInstitutionUserId(null);
                      setSelectedInstitutionId(null);
                      setSelectedInstitutionRole('institution_admin');
                    }
                  } catch (error) {
                    console.error(
                      '❌ Erreur lors de la mise à jour du rôle :',
                      error.response?.data || error.message
                    );
                    alert('⚠️ Impossible de mettre à jour le rôle.');
                  }
                }}
              >
                Valider
              </button>
              <button
                onClick={() => {
                  setShowInstitutionDropdown(false);
                  setPendingInstitutionUserId(null);
                  setSelectedInstitutionId(null);
                  setSelectedInstitutionRole('institution_admin');
                }}
              >
                Annuler
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminUsers;
