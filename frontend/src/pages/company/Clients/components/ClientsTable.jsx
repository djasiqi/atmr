import React from 'react';
import styles from './ClientsTable.module.css';
import ClientTableRowActions from './ClientTableRowActions';
import { getClientDisplayName } from '../../../../utils/clientSearchUtils';

const ClientsTable = ({ clients, onSelect, onEdit, onDelete, selectedClientId, onRefresh: _onRefresh }) => {
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
          {clients.map((client) => {
            const displayName = getClientDisplayName(client);
            const isSelected = selectedClientId === client.id;
            const cityZip = client.domicile?.city
              ? `${client.domicile.zip || ''} ${client.domicile.city}`.trim()
              : client.domicile?.zip || '-';

            return (
            <tr
              key={client.id}
              className={`${!client.is_active ? styles.inactive : ''} ${isSelected ? styles.selected : ''}`}
              onClick={() => onSelect && onSelect(client)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect && onSelect(client);
                }
              }}
              aria-label={`Voir les détails de ${displayName}`}
            >
              <td>
                <div className={styles.clientInfo}>
                  <div className={styles.clientName}>
                    {client.is_institution ? (
                      <>
                        <span className={styles.institutionBadge}>🏢</span>
                        <strong>{displayName}</strong>
                      </>
                    ) : (
                      <>
                        <strong>{displayName}</strong>
                        {/* Indicateurs contextuels */}
                        {client.has_active_stay && (
                          <span className={styles.contextBadge} title="Client hospitalisé">
                            🏥
                          </span>
                        )}
                        {client.has_billing_party && (
                          <span className={styles.contextBadge} title="Tiers payeur configuré">
                            💰
                          </span>
                        )}
                      </>
                    )}
                  </div>
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
                  {client.contact_email ? (
                    <div className={styles.email} title={client.contact_email}>
                      📧 {client.contact_email.length > 25
                        ? `${client.contact_email.substring(0, 22)}...`
                        : client.contact_email}
                    </div>
                  ) : client.contact_phone ? (
                    <div className={styles.phone} title={client.contact_phone}>
                      📞 {client.contact_phone}
                    </div>
                  ) : (
                    <span className={styles.noContact}>-</span>
                  )}
                </div>
              </td>
              <td>
                <div className={styles.address} title={client.domicile?.address || client.billing_address || ''}>
                  {cityZip}
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
              <td onClick={(e) => e.stopPropagation()}>
                <ClientTableRowActions
                  client={client}
                  onEdit={onEdit}
                  onDelete={onDelete}
                  onView={onSelect}
                />
              </td>
            </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default ClientsTable;
