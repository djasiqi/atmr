// pages/institution/Settings/components/TeamsTab.jsx
/**
 * Onglet de gestion des équipes de curateurs (curatelle).
 * Visible uniquement si institution_type === 'curatelle' et rôle admin.
 *
 * Fonctionnalités :
 * - Créer / renommer / supprimer des équipes
 * - Ajouter / retirer des membres (utilisateurs)
 * - Voir le nombre de protégés par équipe
 */

import React, { useState } from 'react';
import { FaPlus, FaTimes, FaTrash, FaEdit, FaUserPlus, FaUsers, FaCheck } from 'react-icons/fa';
import { toast } from 'sonner';
import {
  useInstitutionTeams,
  useCreateTeam,
  useUpdateTeam,
  useDeleteTeam,
  useAddTeamMember,
  useRemoveTeamMember,
  useInstitutionUsers,
} from '../../../../hooks/useInstitutionData';
import styles from '../InstitutionSettings.module.css';

const TeamsTab = () => {
  const { data: teamsData, isLoading } = useInstitutionTeams();
  const { data: usersData } = useInstitutionUsers();
  const createTeamMutation = useCreateTeam();
  const updateTeamMutation = useUpdateTeam();
  const deleteTeamMutation = useDeleteTeam();
  const addMemberMutation = useAddTeamMember();
  const removeMemberMutation = useRemoveTeamMember();

  const [showCreate, setShowCreate] = useState(false);
  const [newTeamName, setNewTeamName] = useState('');
  const [editingTeamId, setEditingTeamId] = useState(null);
  const [editName, setEditName] = useState('');
  const [addingMemberTeamId, setAddingMemberTeamId] = useState(null);
  const [selectedUserId, setSelectedUserId] = useState('');
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);

  const teams = teamsData || [];
  const users = usersData?.users || [];

  const handleCreate = async () => {
    if (!newTeamName.trim()) {
      toast.error('Le nom de l\'équipe est requis');
      return;
    }
    try {
      await createTeamMutation.mutateAsync({ name: newTeamName.trim() });
      setNewTeamName('');
      setShowCreate(false);
      toast.success('Équipe créée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la création');
    }
  };

  const handleRename = async (teamId) => {
    if (!editName.trim()) return;
    try {
      await updateTeamMutation.mutateAsync({ teamId, data: { name: editName.trim() } });
      setEditingTeamId(null);
      toast.success('Équipe renommée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };

  const handleDelete = async (teamId) => {
    try {
      await deleteTeamMutation.mutateAsync(teamId);
      setConfirmDeleteId(null);
      toast.success('Équipe supprimée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };

  const handleAddMember = async (teamId) => {
    if (!selectedUserId) return;
    try {
      await addMemberMutation.mutateAsync({ teamId, userId: parseInt(selectedUserId) });
      setSelectedUserId('');
      toast.success('Membre ajouté');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };

  const handleRemoveMember = async (teamId, userId) => {
    try {
      await removeMemberMutation.mutateAsync({ teamId, userId });
      toast.success('Membre retiré');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };

  const getAvailableUsers = (team) => {
    const memberIds = new Set((team.members || []).map(m => m.user_id));
    return users.filter(u => !memberIds.has(u.id));
  };

  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3>Équipes de curateurs</h3>
        <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
          Organisez vos curateurs en équipes. Chaque équipe ne voit que ses protégés assignés.
          Les administrateurs conservent la visibilité complète.
        </p>
      </div>

      {/* Bouton créer */}
      <div style={{ marginBottom: 20 }}>
        {!showCreate ? (
          <button
            className={styles.addKeyBtn}
            onClick={() => setShowCreate(true)}
            style={{ width: 'auto' }}
          >
            <FaPlus /> Nouvelle équipe
          </button>
        ) : (
          <div style={{
            display: 'flex',
            gap: 8,
            alignItems: 'center',
            padding: '12px 16px',
            background: '#f8f9fa',
            borderRadius: 8,
            border: '1px solid #e0e0e0',
          }}>
            <input
              type="text"
              value={newTeamName}
              onChange={(e) => setNewTeamName(e.target.value)}
              placeholder="Nom de l'équipe"
              onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
              style={{
                flex: 1,
                padding: '8px 12px',
                borderRadius: 6,
                border: '1px solid #ccc',
                fontSize: 13,
              }}
              autoFocus
            />
            <button
              onClick={handleCreate}
              disabled={createTeamMutation.isPending}
              style={{
                padding: '8px 16px',
                borderRadius: 6,
                border: 'none',
                background: '#00796B',
                color: '#fff',
                fontSize: 13,
                cursor: 'pointer',
                fontWeight: 500,
              }}
            >
              {createTeamMutation.isPending ? 'Création...' : 'Créer'}
            </button>
            <button
              onClick={() => { setShowCreate(false); setNewTeamName(''); }}
              style={{
                padding: '8px',
                borderRadius: 6,
                border: '1px solid #ddd',
                background: '#fff',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <FaTimes />
            </button>
          </div>
        )}
      </div>

      {/* Liste des équipes */}
      {isLoading ? (
        <p style={{ color: '#999' }}>Chargement...</p>
      ) : teams.length === 0 ? (
        <div style={{
          background: '#f8f9fa',
          border: '1px solid #e0e0e0',
          borderRadius: 8,
          padding: '24px',
          textAlign: 'center',
          color: '#666',
        }}>
          <FaUsers style={{ fontSize: 24, marginBottom: 8, color: '#aaa' }} />
          <p style={{ margin: 0 }}>Aucune équipe créée.</p>
          <p style={{ margin: '4px 0 0', fontSize: 12, color: '#999' }}>
            Créez des équipes pour organiser vos curateurs et assigner des protégés.
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {teams.map((team) => (
            <div
              key={team.id}
              style={{
                border: '1px solid #e0e0e0',
                borderRadius: 8,
                overflow: 'hidden',
              }}
            >
              {/* Header */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 16px',
                background: '#fafafa',
                borderBottom: '1px solid #e0e0e0',
              }}>
                {editingTeamId === team.id ? (
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', flex: 1 }}>
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleRename(team.id)}
                      style={{
                        flex: 1,
                        padding: '4px 8px',
                        borderRadius: 4,
                        border: '1px solid #ccc',
                        fontSize: 14,
                      }}
                      autoFocus
                    />
                    <button
                      onClick={() => handleRename(team.id)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#00796B' }}
                    >
                      <FaCheck />
                    </button>
                    <button
                      onClick={() => setEditingTeamId(null)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#999' }}
                    >
                      <FaTimes />
                    </button>
                  </div>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <FaUsers style={{ color: '#7C3AED' }} />
                    <strong style={{ fontSize: 14 }}>{team.name}</strong>
                    <span style={{
                      fontSize: 11,
                      color: '#888',
                      background: '#f0f0f0',
                      padding: '2px 8px',
                      borderRadius: 10,
                    }}>
                      {team.members_count || 0} membre{(team.members_count || 0) !== 1 ? 's' : ''}
                    </span>
                    <span style={{
                      fontSize: 11,
                      color: '#7C3AED',
                      background: '#f5f0ff',
                      padding: '2px 8px',
                      borderRadius: 10,
                    }}>
                      {team.patients_count || 0} protégé{(team.patients_count || 0) !== 1 ? 's' : ''}
                    </span>
                  </div>
                )}

                {editingTeamId !== team.id && (
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      onClick={() => { setEditingTeamId(team.id); setEditName(team.name); }}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#666', padding: 4 }}
                      title="Renommer"
                    >
                      <FaEdit />
                    </button>
                    <button
                      onClick={() => setConfirmDeleteId(team.id)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#e53935', padding: 4 }}
                      title="Supprimer"
                    >
                      <FaTrash />
                    </button>
                  </div>
                )}
              </div>

              {/* Membres */}
              <div style={{ padding: '12px 16px' }}>
                {(team.members || []).length === 0 ? (
                  <p style={{ fontSize: 12, color: '#999', margin: '0 0 8px', fontStyle: 'italic' }}>
                    Aucun membre dans cette équipe.
                  </p>
                ) : (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
                    {(team.members || []).map((member) => (
                      <span
                        key={member.id}
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 6,
                          padding: '4px 10px',
                          background: '#e8f5e9',
                          borderRadius: 14,
                          fontSize: 12,
                          color: '#2e7d32',
                          fontWeight: 500,
                        }}
                      >
                        {member.user_name || member.user_email}
                        <button
                          onClick={() => handleRemoveMember(team.id, member.user_id)}
                          style={{
                            background: 'none',
                            border: 'none',
                            cursor: 'pointer',
                            color: '#2e7d32',
                            padding: 0,
                            fontSize: 12,
                            lineHeight: 1,
                          }}
                          title="Retirer"
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                )}

                {/* Ajouter un membre */}
                {addingMemberTeamId === team.id ? (
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <select
                      value={selectedUserId}
                      onChange={(e) => setSelectedUserId(e.target.value)}
                      style={{
                        flex: 1,
                        padding: '6px 8px',
                        borderRadius: 6,
                        border: '1px solid #ccc',
                        fontSize: 13,
                      }}
                    >
                      <option value="">Sélectionner un utilisateur...</option>
                      {getAvailableUsers(team).map((u) => (
                        <option key={u.id} value={u.id}>
                          {u.first_name} {u.last_name} ({u.email})
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={() => handleAddMember(team.id)}
                      disabled={!selectedUserId || addMemberMutation.isPending}
                      style={{
                        padding: '6px 12px',
                        borderRadius: 6,
                        border: 'none',
                        background: selectedUserId ? '#00796B' : '#ccc',
                        color: '#fff',
                        fontSize: 12,
                        cursor: selectedUserId ? 'pointer' : 'default',
                        fontWeight: 500,
                      }}
                    >
                      Ajouter
                    </button>
                    <button
                      onClick={() => { setAddingMemberTeamId(null); setSelectedUserId(''); }}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#999' }}
                    >
                      <FaTimes />
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setAddingMemberTeamId(team.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '4px 10px',
                      background: 'none',
                      border: '1px dashed #ccc',
                      borderRadius: 14,
                      fontSize: 12,
                      color: '#888',
                      cursor: 'pointer',
                    }}
                  >
                    <FaUserPlus /> Ajouter un membre
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal confirmation suppression */}
      {confirmDeleteId && (
        <div className={styles.modal}>
          <div className={styles.modalContent}>
            <div className={styles.modalHeader}>
              <h3>Supprimer cette équipe ?</h3>
              <button onClick={() => setConfirmDeleteId(null)}><FaTimes /></button>
            </div>
            <div className={styles.modalBody}>
              <p style={{ margin: 0, color: '#666', lineHeight: 1.6 }}>
                Les protégés assignés à cette équipe seront désassignés mais pas supprimés.
                Les membres ne perdent pas leur accès à l'institution.
              </p>
            </div>
            <div className={styles.modalActions}>
              <button onClick={() => setConfirmDeleteId(null)}>Annuler</button>
              <button
                className={styles.revokeBtn}
                onClick={() => handleDelete(confirmDeleteId)}
                disabled={deleteTeamMutation.isPending}
                style={{ padding: '10px 20px', borderRadius: '8px', fontSize: '14px' }}
              >
                {deleteTeamMutation.isPending ? 'Suppression...' : 'Confirmer la suppression'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeamsTab;
