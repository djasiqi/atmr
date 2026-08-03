import React, { useCallback, useEffect, useState } from 'react';
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
import AdminActionDialog from '../components/AdminActionDialog';
import AdminTempPasswordDialog from '../components/AdminTempPasswordDialog';
import styles from './AdminUsers.module.css';
import adminShell from '../adminShell.module.css';
import { toast } from 'sonner';

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
  const [loadError, setLoadError] = useState(null);

  const [companyOptions, setCompanyOptions] = useState([]);
  const [showCompanyDropdown, setShowCompanyDropdown] = useState(false);
  const [pendingDriverUserId, setPendingDriverUserId] = useState(null);

  const [institutionOptions, setInstitutionOptions] = useState([]);
  const [showInstitutionDropdown, setShowInstitutionDropdown] = useState(false);
  const [pendingInstitutionUserId, setPendingInstitutionUserId] = useState(null);
  const [selectedInstitutionId, setSelectedInstitutionId] = useState(null);
  const [selectedInstitutionRole, setSelectedInstitutionRole] = useState('institution_admin');

  const [actionDialog, setActionDialog] = useState(null);
  const [tempPasswordDialog, setTempPasswordDialog] = useState(null);

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
    setLoadError(null);
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
      const status = error?.response?.status;
      let message = 'Impossible de charger les utilisateurs.';
      if (status === 401 || status === 403) {
        message = 'Accès refusé — vérifiez vos droits administrateur.';
      } else if (status >= 500) {
        message = 'Service indisponible — réessayez plus tard.';
      } else if (!error?.response) {
        message = 'Erreur réseau — vérifiez votre connexion.';
      }
      setLoadError(message);
      setUsers([]);
      setTotalUsers(0);
      setTotalPages(1);
    } finally {
      setLoading(false);
    }
  }, [debouncedSearch, page, perPage, roleFilter, sortBy, sortOrder]);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const updateUserRoleHandler = async (userId, newRole) => {
    if (!userId || !newRole) {
      toast.error("L'utilisateur ou le rôle est invalide.");
      return;
    }

    if (newRole.toLowerCase() === 'driver') {
      if (!companyOptions.length) {
        toast.error('Aucune entreprise disponible.');
        return;
      }
      setPendingDriverUserId(userId);
      setShowCompanyDropdown(true);
      return;
    }

    if (newRole.toLowerCase() === 'institution') {
      setActionDialog({
        kind: 'role-institution',
        userId,
        title: 'Attribuer le rôle institution',
        description: 'Confirmer l’attribution du rôle institution (admin institution).',
        confirmationLabel: 'Attribuer',
        onConfirm: async () => {
          await updateUserRole(userId, {
            role: 'institution',
            institution_role: 'institution_admin',
          });
          await loadUsers();
          setActionDialog(null);
          toast.success('Rôle institution attribué.');
        },
      });
      return;
    }

    setActionDialog({
      kind: 'role',
      userId,
      title: 'Changer le rôle',
      description: `Nouveau rôle : ${ROLE_LABELS[newRole.toLowerCase()] || newRole}.`,
      confirmationLabel: 'Mettre à jour',
      onConfirm: async () => {
        await updateUserRole(userId, { role: newRole });
        await loadUsers();
        setActionDialog(null);
        toast.success(`Rôle mis à jour : ${newRole}`);
      },
    });
  };

  const handleDelete = (user) => {
    setActionDialog({
      kind: 'delete',
      title: 'Supprimer l’utilisateur',
      description: `Supprimer définitivement le compte « ${user.username || user.email} ».`,
      impact: 'Cette action est irréversible.',
      confirmationLabel: 'Supprimer',
      confirmText: 'SUPPRIMER',
      danger: true,
      onConfirm: async () => {
        await deleteUser(user.id);
        await loadUsers();
        setActionDialog(null);
        toast.success('Utilisateur supprimé.');
      },
    });
  };

  const handleResetPassword = (user) => {
    setActionDialog({
      kind: 'reset-password',
      title: 'Réinitialiser le mot de passe',
      description: `Générer un mot de passe temporaire pour « ${user.username || user.email} ».`,
      impact: 'Le mot de passe actuel ne fonctionnera plus.',
      confirmationLabel: 'Réinitialiser',
      danger: true,
      onConfirm: async () => {
        const response = await resetUserPassword(user.id);
        setActionDialog(null);
        if (!response?.new_password) {
          throw new Error('Aucun mot de passe généré par le serveur.');
        }
        setTempPasswordDialog({
          accountLabel: user.email || user.username || `ID ${user.id}`,
          temporaryPassword: response.new_password,
        });
      },
    });
  };

  const handleSetBillingAccess = (user, state) => {
    if (!user?.company_id) {
      toast.error('Aucune entreprise liée à ce compte.');
      return;
    }
    const label = formatBillingAccessLabel(state);
    const needsPauseDays = state === 'active';
    setActionDialog({
      kind: 'billing-access',
      title: 'Accès commercial',
      description:
        state === 'active'
          ? 'Lever la restriction d’accès commercial pour cette entreprise.'
          : `Appliquer une restriction d’accès commercial (${label}).`,
      confirmationLabel: 'Confirmer',
      reason: needsPauseDays
        ? {
            required: false,
            label: 'Pause du recouvrement après levée (jours, vide = aucune)',
          }
        : undefined,
      onConfirm: async ({ reason }) => {
        const payload = {
          state,
          reason_code: state === 'active' ? 'admin_lift' : 'admin_manual',
        };
        if (needsPauseDays && reason && String(reason).trim() !== '') {
          const pauseDays = parseInt(reason, 10);
          if (!Number.isNaN(pauseDays) && pauseDays > 0) {
            payload.pause_days_after_lift = pauseDays;
          }
        }
        await setCompanyBillingAccess(user.company_id, payload);
        await loadUsers();
        setActionDialog(null);
        toast.success(`Accès commercial mis à jour : ${label}`);
      },
    });
  };

  const handlePauseDunning = (user) => {
    if (!user?.company_id) {
      toast.error('Aucune entreprise liée à ce compte.');
      return;
    }
    setActionDialog({
      kind: 'pause-dunning',
      title: 'Pause du recouvrement',
      description: 'Mettre en pause le recouvrement automatique pour cette entreprise.',
      confirmationLabel: 'Mettre en pause',
      reason: {
        required: true,
        label: 'Durée de pause (jours)',
        minLength: 1,
      },
      onConfirm: async ({ reason }) => {
        const days = parseInt(reason, 10);
        if (Number.isNaN(days) || days < 1) {
          throw new Error('Durée invalide — indiquez un nombre de jours ≥ 1.');
        }
        await pauseCompanyDunning(user.company_id, {
          days,
          reason: 'pause_admin',
        });
        await loadUsers();
        setActionDialog(null);
        toast.success(`Recouvrement mis en pause pour ${days} jour(s).`);
      },
    });
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
                ) : loadError ? (
                  <tr>
                    <td colSpan="6" className={styles.placeholderRow} role="alert">
                      {loadError}{' '}
                      <button type="button" className={styles.ghostFilterButton} onClick={loadUsers}>
                        Réessayer
                      </button>
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
                            onClick={() => handleResetPassword(user)}
                            className={styles.resetButton}
                          >
                            Reinitialiser
                          </button>
                          <button
                            onClick={() => handleDelete(user)}
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
                      Aucun utilisateur trouvé
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
                  toast.error('Veuillez sélectionner une entreprise.');
                  return;
                }
                try {
                  await updateUserRole(pendingDriverUserId, {
                    role: 'driver',
                    company_id: selectedCompany.id,
                  });
                  await loadUsers();
                  setShowCompanyDropdown(false);
                  setPendingDriverUserId(null);
                  toast.success('Rôle chauffeur attribué.');
                } catch (error) {
                  toast.error(
                    error?.response?.data?.error ||
                      'Impossible de mettre à jour le rôle.'
                  );
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
                    toast.error('Veuillez sélectionner une institution.');
                    return;
                  }
                  try {
                    await updateUserRole(pendingInstitutionUserId, {
                      role: 'institution',
                      institution_id: selectedInstitutionId,
                      institution_role: selectedInstitutionRole,
                    });
                    await loadUsers();
                    setShowInstitutionDropdown(false);
                    setPendingInstitutionUserId(null);
                    setSelectedInstitutionId(null);
                    setSelectedInstitutionRole('institution_admin');
                    toast.success('Rôle institution attribué.');
                  } catch (error) {
                    toast.error(
                      error?.response?.data?.error ||
                        'Impossible de mettre à jour le rôle.'
                    );
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

      {actionDialog ? (
        <AdminActionDialog
          open
          title={actionDialog.title}
          description={actionDialog.description}
          impact={actionDialog.impact}
          confirmationLabel={actionDialog.confirmationLabel}
          confirmText={actionDialog.confirmText}
          reason={actionDialog.reason}
          danger={Boolean(actionDialog.danger)}
          onConfirm={actionDialog.onConfirm}
          onClose={() => setActionDialog(null)}
        />
      ) : null}

      {tempPasswordDialog ? (
        <AdminTempPasswordDialog
          open
          accountLabel={tempPasswordDialog.accountLabel}
          temporaryPassword={tempPasswordDialog.temporaryPassword}
          onClose={() => setTempPasswordDialog(null)}
        />
      ) : null}
    </>
  );
};

export default AdminUsers;
