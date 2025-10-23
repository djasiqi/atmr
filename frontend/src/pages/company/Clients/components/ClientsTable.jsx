import React from 'react';
import styles from './ClientsTable.module.css';

const ClientsTable = ({ clients, onEdit, onDelete, onRefresh: _onRefresh }) => {
  if (!clients || clients.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>👥</div>
        <h3>Aucun client trouvé</h3>
        <p>Créez votre premier client pour commencer</p>
      </div>
    );
  }

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      return new Date(dateString).toLocaleDateString('fr-FR');
    } catch {
      return '-';
    }
  };

  return (
    <div className={styles.tableContainer}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Client</th>
            <th>Type</th>
            <th>Contact</th>
            <th>Adresse</th>
            <th>Statut</th>
            <th>Date création</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {clients.map((client) => (
            <tr key={client.id} className={!client.is_active ? styles.inactive : ''}>
              <td>
                <div className={styles.clientInfo}>
                  <div className={styles.clientName}>
                    {client.is_institution ? (
                      <>
                        <span className={styles.institutionBadge}>🏥</span>
                        <strong>{client.institution_name || 'Institution'}</strong>
                      </>
                    ) : (
                      <strong>
                        {client.first_name || ''} {client.last_name || ''}
                      </strong>
                    )}
                  </div>
                  {!client.is_institution && client.institution_name && (
                    <div className={styles.clientSubInfo}>{client.institution_name}</div>
                  )}
                </div>
              </td>
              <td>
                <span
                  className={`${styles.typeBadge} ${
                    client.is_institution ? styles.institution : styles.regular
                  }`}
                >
                  {client.is_institution ? 'Institution' : 'Client'}
                </span>
              </td>
              <td>
                <div className={styles.contactInfo}>
                  {client.contact_email && (
                    <div className={styles.email}>📧 {client.contact_email}</div>
                  )}
                  {client.contact_phone && (
                    <div className={styles.phone}>📞 {client.contact_phone}</div>
                  )}
                  {!client.contact_email && !client.contact_phone && (
                    <span className={styles.noContact}>-</span>
                  )}
                </div>
              </td>
              <td>
                <div className={styles.address}>
                  {client.domicile?.address
                    ? `${client.domicile.address}${
                        client.domicile.zip ? ', ' + client.domicile.zip : ''
                      }${client.domicile.city ? ', ' + client.domicile.city : ''}`
                    : client.billing_address || '-'}
                </div>
              </td>
              <td>
                <span
                  className={`${styles.statusBadge} ${
                    client.is_active ? styles.active : styles.inactive
                  }`}
                >
                  {client.is_active ? 'Actif' : 'Inactif'}
                </span>
              </td>
              <td>{formatDate(client.created_at)}</td>
              <td>
                <div className={styles.actions}>
                  <button
                    onClick={() => onEdit(client)}
                    className={`btn btn-sm btn-primary ${styles.editBtn}`}
                    title="Éditer le client"
                  >
                    ✏️ Éditer
                  </button>
                  <button
                    onClick={() => onDelete(client)}
                    className={`btn btn-sm ${styles.deleteBtn}`}
                    title="Supprimer le client"
                  >
                    🗑️
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ClientsTable;
