import React, { useCallback, useEffect, useState } from 'react';
import apiClient from '../../../utils/apiClient';
import {
  fetchUsers,
  deleteUser,
  resetUserPassword,
  updateUserRole,
  fetchCompanies,
  fetchInstitutions,
  setCompanyBillingAccess,
  pauseCompanyDunning,
} from '../../../services/adminService';
import styles from './AdminUsers.module.css';
import adminShell from '../adminShell.module.css';

const ROLE_LABELS = {
  admin: 'Admin',
  client: 'Client',
  driver: 'Chauffeur',
  company: 'Entreprise',
  institution: 'Institution',
};

const BILLING_ACCESS_LABELS = {
  active: 'Actif',
  partial: 'Partiel',
  full: 'Complet',
};

const formatBillingAccessLabel = (state) =>
  BILLING_ACCESS_LABELS[String(state || 'active').toLowerCase()] || 'Actif';

const billingAccessBadgeClass = (state, stylesMap) => {
  const key = String(state || 'active').toLowerCase();
  if (key === 'partial') return stylesMap.billingBadgePartial;
  if (key === 'full') return stylesMap.billingBadgeFull;
  return stylesMap.billingBadgeActive;
};

const AdminUsers = () => {
  const [users, setUsers] = useState([]);
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc');
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState(50);
  const [totalUsers, setTotalUsers] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [globalStats, setGlobalStats] = useState({
    admin: 0,
    company: 0,
    institution: 0,
    driver: 0,
    client: 0,
  });
  const [loading, setLoading] = useState(true);

  const [companyOptions, setCompanyOptions] = useState([]);
  const [showCompanyDropdown, setShowCompanyDropdown] = useState(false);
  const [pendingDriverUserId, setPendingDriverUserId] = useState(null);

  const [institutionOptions, setInstitutionOptions] = useState([]);
  const [showInstitutionDropdown, setShowInstitutionDropdown] = useState(false);
  const [pendingInstitutionUserId, setPendingInstitutionUserId] = useState(null);
  const [selectedInstitutionId, setSelectedInstitutionId] = useState(null);
  const [selectedInstitutionRole, setSelectedInstitutionRole] = useState('institution_admin');

  useEffect(() => {
    const loadCompanies = async () => {
      try {
        const companies = await fetchCompanies();
        setCompanyOptions((companies || []).map((c) => ({ ...c, selected: false })));
      } catch (error) {
        console.error('Erreur chargement entreprises :', error);
      }
    };
    loadCompanies();
  }, []);

  useEffect(() => {
    const loadInstitutions = async () => {
      try {
        const institutions = await fetchInstitutions();
        setInstitutionOptions(institutions || []);
      } catch (error) {
        console.error('Erreur chargement institutions :', error);
      }
    };
    loadInstitutions();
  }, []);

  useEffect(() => {
    const timeout = setTimeout(() => {
      setDebouncedSearch(search.trim());
      setPage(1);
    }, 300);
    return () => clearTimeout(timeout);
  }, [search]);

  const loadUsers = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchUsers({
        page,
        per_page: perPage,
        search: debouncedSearch,
        role: roleFilter,
        sort_by: sortBy,
        sort_order: sortOrder,
      });
      setUsers(data.users || []);
      setTotalUsers(data.total || 0);
      setTotalPages(data.total_pages || 1);
      if (data.role_counts) {
        setGlobalStats({
          admin: Number(data.role_counts.admin || 0),
          company: Number(data.role_counts.company || 0),
          institution: Number(data.role_counts.institution || 0),
          driver: Number(data.role_counts.driver || 0),
          client: Number(data.role_counts.client || 0),
        });
      } else {
        setGlobalStats({
          admin: 0,
          company: 0,
          institution: 0,
          driver: 0,
          client: 0,
        });
      }

      if ((data.total_pages || 1) < page) {
        setPage(1);
      }
    } catch (error) {
      console.error('Erreur chargement utilisateurs :', error);
      setUsers([]);
      setTotalUsers(0);
      setTotalPages(1);
      setGlobalStats({
        admin: 0,
        company: 0,
        institution: 0,
        driver: 0,
        client: 0,
      });
    } finally {
      setLoading(false);
    }
  }, [debouncedSearch, page, perPage, roleFilter, sortBy, sortOrder]);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const updateUserRoleHandler = async (userId, newRole) => {
    if (!userId || !newRole) {
      alert("Erreur : l utilisateur ou le role est invalide.");
      return;
    }

    if (newRole.toLowerCase() === 'driver') {
      if (!companyOptions.length) {
        alert('Aucune entreprise disponible.');
        return;
      }

      setPendingDriverUserId(userId);
      setShowCompanyDropdown(true);
    } else if (newRole.toLowerCase() === 'institution') {
      try {
        await updateUserRole(userId, {
          role: 'institution',
          institution_role: 'institution_admin',
        });
        alert('Role institution attribue avec succes.');
        await loadUsers();
      } catch (error) {
        console.error('Erreur attribution role institution :', error);
        alert('Impossible d attribuer le role institution.');
      }
      return;
    }

    try {
      await updateUserRole(userId, { role: newRole });
      alert(`Role mis a jour avec succes : ${newRole}`);
      await loadUsers();
    } catch (error) {
      console.error('Erreur mise a jour role :', error);
      alert('Impossible de mettre a jour le role.');
    }
  };

  const handleDelete = async (userId) => {
    if (!window.confirm('Confirmer la suppression de cet utilisateur ?')) {
      return;
    }
    try {
      await deleteUser(userId);
      await loadUsers();
      alert('Utilisateur supprime avec succes.');
    } catch (error) {
      console.error('Erreur suppression utilisateur :', error);
      alert("Impossible de supprimer l utilisateur.");
    }
  };

  const handleResetPassword = async (userId) => {
    if (!userId) {
      console.error('Erreur : userId est undefined.');
      alert('Impossible de reinitialiser le mot de passe : ID utilisateur introuvable.');
      return;
    }

    const confirmation = window.confirm(
      'Voulez-vous vraiment réinitialiser le mot de passe de cet utilisateur ?'
    );

    if (!confirmation) return;

    try {
      const response = await resetUserPassword(userId);

      if (response?.new_password) {
        alert(`Mot de passe reinitialise : ${response.new_password}`);
      } else {
        console.warn('La reponse API ne contient pas de mot de passe.');
        alert('Echec de la reinitialisation : aucun mot de passe genere.');
      }
    } catch (error) {
      console.error(
        'Erreur lors de la reinitialisation du mot de passe :',
        error.response?.data || error.message
      );
      alert('Une erreur est survenue lors de la reinitialisation.');
    }
  };

  const handleSetBillingAccess = async (user, state) => {
    if (!user?.company_id) {
      alert("Aucune entreprise liee a ce compte.");
      return;
    }
    const label = formatBillingAccessLabel(state);
    const confirmMsg =
      state === 'active'
        ? 'Lever la restriction d acces commercial pour cette entreprise ?'
        : `Appliquer une restriction d acces commercial (${label}) pour cette entreprise ?`;
    if (!window.confirm(confirmMsg)) {
      return;
    }
    try {
      const payload = {
        state,
        reason_code: state === 'active' ? 'admin_lift' : 'admin_manual',
      };
      if (state === 'active') {
        const pauseDaysRaw = window.prompt(
          'Pause du recouvrement automatique apres levee (jours, laisser vide pour aucune) :',
          '14'
        );
        if (pauseDaysRaw !== null && String(pauseDaysRaw).trim() !== '') {
          const pauseDays = parseInt(pauseDaysRaw, 10);
          if (!Number.isNaN(pauseDays) && pauseDays > 0) {
            payload.pause_days_after_lift = pauseDays;
          }
        }
      }
      await setCompanyBillingAccess(user.company_id, payload);
      await loadUsers();
      alert(`Acces commercial mis a jour : ${label}`);
    } catch (error) {
      console.error('Erreur mise a jour acces billing :', error.response?.data || error.message);
      alert("Impossible de mettre a jour l acces commercial.");
    }
  };

  const handlePauseDunning = async (user) => {
    if (!user?.company_id) {
      alert("Aucune entreprise liee a ce compte.");
      return;
    }
    const daysRaw = window.prompt('Duree de pause du recouvrement (jours) :', '14');
    if (daysRaw === null) {
      return;
    }
    const days = parseInt(daysRaw, 10);
    if (Number.isNaN(days) || days < 1) {
      alert('Duree invalide.');
      return;
    }
    try {
      await pauseCompanyDunning(user.company_id, {
        days,
        reason: 'pause_admin',
      });
      await loadUsers();
      alert(`Recouvrement mis en pause pour ${days} jour(s).`);
    } catch (error) {
      console.error('Erreur pause dunning :', error.response?.data || error.message);
      alert('Impossible de mettre en pause le recouvrement.');
    }
  };

  const startRow = totalUsers === 0 ? 0 : (page - 1) * perPage + 1;
  const endRow = Math.min(page * perPage, totalUsers);

  const resetFilters = () => {
    setSearch('');
    setRoleFilter('');
    setSortBy('created_at');
    setSortOrder('desc');
    setPage(1);
  };

  return (
    <>
      <main className={`${adminShell.content} ${styles.shellMain}`}>
          <header className={styles.pageHeader}>
            <h1>Gestion des utilisateurs</h1>
            <p className={styles.subtext}>
              Recherche, tri, attribution des roles, acces commercial entreprises et
              actions de maintenance des comptes.
            </p>
          </header>

          <section className={styles.metricsGrid} aria-label="Synthese utilisateurs">
            <article className={styles.metricCard}>
              <span>Total</span>
              <strong>{totalUsers}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>Admins</span>
              <strong>{globalStats.admin}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>Entreprises</span>
              <strong>{globalStats.company}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>Institutions</span>
              <strong>{globalStats.institution}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>Chauffeurs</span>
              <strong>{globalStats.driver}</strong>
            </article>
            <article className={styles.metricCard}>
              <span>Clients</span>
              <strong>{globalStats.client}</strong>
            </article>
          </section>

          <section className={styles.toolbar}>
            <input
              type="text"
              placeholder="Rechercher par nom ou email"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className={styles.searchInput}
            />

            <div className={styles.filters}>
              <select
                value={roleFilter}
                onChange={(e) => {
                  setRoleFilter(e.target.value);
                  setPage(1);
                }}
                className={styles.roleFilter}
              >
                <option value="">Tous les roles</option>
                <option value="admin">Admin</option>
                <option value="client">Client</option>
                <option value="driver">Chauffeur</option>
                <option value="company">Entreprise</option>
                <option value="institution">Institution</option>
              </select>

              <select
                value={sortBy}
                onChange={(e) => {
                  setSortBy(e.target.value);
                  setPage(1);
                }}
                className={styles.roleFilter}
              >
                <option value="created_at">Tri : date inscription</option>
                <option value="username">Tri : nom</option>
                <option value="role">Tri : role</option>
                <option value="email">Tri : email</option>
              </select>

              <select
                value={sortOrder}
                onChange={(e) => {
                  setSortOrder(e.target.value);
                  setPage(1);
                }}
                className={styles.roleFilter}
              >
                <option value="desc">Ordre : decroissant</option>
                <option value="asc">Ordre : croissant</option>
              </select>

              <button type="button" className={styles.ghostFilterButton} onClick={resetFilters}>
                Reinitialiser filtres
              </button>
            </div>
          </section>

          <div className={styles.tableContainer}>
            <table className={styles.userTable}>
              <thead>
                <tr>
                  <th>Nom</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th>Acces commercial</th>
                  <th>Date d inscription</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan="6" className={styles.placeholderRow}>
                      Chargement des utilisateurs...
                    </td>
                  </tr>
                ) : users.length > 0 ? (
                  users.map((user) => {
                    const userRole = String(user.role || '').toLowerCase();
                    const isCompany = userRole === 'company' && Boolean(user.company_id);
                    const billingState = String(
                      user.platform_billing_access_state || 'active'
                    ).toLowerCase();
                    const dunningPaused = Boolean(user.dunning_paused_until);
                    return (
                      <tr key={user.id}>
                        <td className={styles.userNameCell}>
                          {user.username || '-'}
                          {isCompany && user.company_name ? (
                            <span className={styles.companyHint}>{user.company_name}</span>
                          ) : null}
                        </td>
                        <td className={styles.emailCell}>{user.email || '-'}</td>
                        <td>
                          <select
                            className={styles.roleSelect}
                            value={userRole}
                            onChange={(e) => updateUserRoleHandler(user.id, e.target.value)}
                          >
                            {Object.entries(ROLE_LABELS).map(([value, label]) => (
                              <option key={value} value={value}>
                                {label}
                              </option>
                            ))}
                          </select>
                        </td>
                        <td className={styles.billingAccessCell}>
                          {isCompany ? (
                            <div className={styles.billingAccessBlock}>
                              <span
                                className={`${styles.billingBadge} ${billingAccessBadgeClass(
                                  billingState,
                                  styles
                                )}`}
                              >
                                {formatBillingAccessLabel(billingState)}
                              </span>
                              {dunningPaused ? (
                                <span className={styles.dunningPauseHint}>
                                  Pause jusqu au{' '}
                                  {new Date(user.dunning_paused_until).toLocaleDateString(
                                    'fr-CH'
                                  )}
                                </span>
                              ) : null}
                              <div className={styles.billingActions}>
                                {billingState !== 'active' ? (
                                  <button
                                    type="button"
                                    className={styles.billingActionButton}
                                    onClick={() => handleSetBillingAccess(user, 'active')}
                                  >
                                    Lever
                                  </button>
                                ) : null}
                                {billingState !== 'partial' ? (
                                  <button
                                    type="button"
                                    className={styles.billingActionButton}
                                    onClick={() => handleSetBillingAccess(user, 'partial')}
                                  >
                                    Partiel
                                  </button>
                                ) : null}
                                {billingState !== 'full' ? (
                                  <button
                                    type="button"
                                    className={styles.billingActionButtonWarn}
                                    onClick={() => handleSetBillingAccess(user, 'full')}
                                  >
                                    Complet
                                  </button>
                                ) : null}
                                <button
                                  type="button"
                                  className={styles.billingActionButton}
                                  onClick={() => handlePauseDunning(user)}
                                >
                                  Pause
                                </button>
                              </div>
                            </div>
                          ) : (
                            <span className={styles.billingNa}>—</span>
                          )}
                        </td>
                        <td className={styles.dateCell}>
                          {user.created_at
                            ? new Date(user.created_at).toLocaleString('fr-CH')
                            : 'Inconnu'}
                        </td>
                        <td className={styles.actionsCell}>
                          <button
                            onClick={() => handleResetPassword(user.id)}
                            className={styles.resetButton}
                          >
                            Reinitialiser
                          </button>
                          <button
                            onClick={() => handleDelete(user.id)}
                            className={styles.deleteButton}
                          >
                            Supprimer
                          </button>
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan="6" className={styles.placeholderRow}>
                      Aucun utilisateur trouve
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className={styles.paginationBar}>
            <p className={styles.paginationInfo}>
              Affichage {startRow}-{endRow} sur {totalUsers}
            </p>
            <div className={styles.paginationControls}>
              <label className={styles.perPageLabel}>
                Lignes
                <select
                  value={perPage}
                  onChange={(e) => {
                    setPerPage(parseInt(e.target.value, 10));
                    setPage(1);
                  }}
                  className={styles.perPageSelect}
                >
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                </select>
              </label>
              <button
                type="button"
                className={styles.paginationButton}
                disabled={page <= 1 || loading}
                onClick={() => setPage((prev) => Math.max(1, prev - 1))}
              >
                Precedent
              </button>
              <span className={styles.pageIndicator}>
                Page {page} / {Math.max(totalPages, 1)}
              </span>
              <button
                type="button"
                className={styles.paginationButton}
                disabled={page >= totalPages || loading}
                onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              >
                Suivant
              </button>
            </div>
          </div>
        </main>
      {showCompanyDropdown && (
        <div className={styles.modal}>
          <div className={styles.modalContent}>
            <h3>Assigner une entreprise au chauffeur</h3>
            <select
              className={styles.modalInput}
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
              className={styles.modalPrimaryButton}
              onClick={async () => {
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
                  );
                  if (response.status === 200) {
                    alert('Role mis a jour avec succes : driver');
                    await loadUsers();
                    setShowCompanyDropdown(false);
                    setPendingDriverUserId(null);
                  }
                } catch (error) {
                  console.error(
                    'Erreur lors de la mise a jour du role :',
                    error.response?.data || error.message
                  );
                  alert('Impossible de mettre a jour le role.');
                }
              }}
            >
              Valider
            </button>
            <button
              className={styles.modalGhostButton}
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

      {showInstitutionDropdown && (
        <div className={styles.modal}>
          <div className={styles.modalContent}>
            <h3>Assigner une institution</h3>
            <div className={styles.modalField}>
              <label>Institution</label>
              <select
                className={styles.modalInput}
                value={selectedInstitutionId || ''}
                onChange={(e) => setSelectedInstitutionId(parseInt(e.target.value, 10) || null)}
              >
                <option value="">Sélectionnez une institution</option>
                {institutionOptions.map((inst) => (
                  <option key={inst.id} value={inst.id}>
                    {inst.name} ({inst.type || 'clinique'})
                  </option>
                ))}
              </select>
            </div>
            <div className={styles.modalField}>
              <label>Role dans l institution</label>
              <select
                className={styles.modalInput}
                value={selectedInstitutionRole}
                onChange={(e) => setSelectedInstitutionRole(e.target.value)}
              >
                <option value="institution_admin">Admin institution</option>
                <option value="institution_requester">Demandeur</option>
                <option value="institution_reader">Lecteur</option>
                <option value="institution_billing">Facturation</option>
              </select>
            </div>
            <div className={styles.modalActions}>
              <button
                className={styles.modalPrimaryButton}
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
                      alert('Role mis a jour avec succes : institution');
                      await loadUsers();
                      setShowInstitutionDropdown(false);
                      setPendingInstitutionUserId(null);
                      setSelectedInstitutionId(null);
                      setSelectedInstitutionRole('institution_admin');
                    }
                  } catch (error) {
                    console.error(
                      'Erreur lors de la mise a jour du role :',
                      error.response?.data || error.message
                    );
                    alert('Impossible de mettre a jour le role.');
                  }
                }}
              >
                Valider
              </button>
              <button
                className={styles.modalGhostButton}
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
    </>
  );
};

export default AdminUsers;
