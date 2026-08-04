import React from 'react';
import { FiHome, FiPhone, FiMail, FiInbox } from 'react-icons/fi';
import styles from './ClientsTable.module.css';
import ClientTableRowActions from './ClientTableRowActions';
import { getClientDisplayName } from '../../../../utils/clientSearchUtils';

const ClientsTable = ({
  clients,
  onSelect,
  onEdit,
  onDelete,
  selectedClientId,
  onRefresh: _onRefresh,
  compact = false,
}) => {
  if (!clients || clients.length === 0) {
    return (
      <div className={styles.empty}>
        <FiInbox size={48} className={styles.emptyIcon} />
        <h3>Aucun client trouvé</h3>
        <p>Modifiez vos filtres ou ajoutez un client</p>
      </div>
    );
  }

  const formatDate = (dateString) => {
    if (!dateString) return '\u2014';
    try {
      return new Date(dateString).toLocaleDateString('fr-FR');
    } catch {
      return '\u2014';
    }
  };

  const getContactDisplay = (client) => {
    if (client.contact_phone) {
      return { value: client.contact_phone, type: 'phone' };
    }
    if (client.contact_email) {
      return { value: client.contact_email, type: 'email' };
    }
    return { value: '\u2014', type: null };
  };

  const getAddressDisplay = (client) => {
    const dom = client.domicile;
    if (dom?.city && dom?.zip) {
      const short = `${dom.zip} ${dom.city}`;
      const full = dom.address ? `${dom.address}, ${short}` : short;
      return { short, full };
    }
    if (dom?.address) {
      return { short: dom.address, full: dom.address };
    }
    return { short: '\u2014', full: '' };
  };

  return (
    <div
      className={`${styles.tableContainer} ${compact ? styles.tableCompact : ''}`}
    >
      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.colClient}>Client</th>
              <th className={styles.colContact}>Contact</th>
              <th className={`${styles.colAddress} ${styles.colSecondary}`}>Adresse</th>
              <th className={styles.colStatus}>Statut</th>
              <th className={`${styles.colDate} ${styles.colSecondary}`}>Créé le</th>
              <th className={styles.colActionsHead}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {clients.map((client) => {
            const displayName = getClientDisplayName(client);
            const isSelected = selectedClientId === client.id;
            const contact = getContactDisplay(client);
            const address = getAddressDisplay(client);

            return (
              <tr
                key={client.id}
                className={`${!client.is_active ? styles.rowInactive : ''} ${isSelected ? styles.selected : ''}`}
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
                aria-current={isSelected ? 'true' : undefined}
              >
                <td>
                  <div className={styles.clientName}>
                    {client.is_institution && (
                      <FiHome size={12} className={styles.institutionIcon} />
                    )}
                    <strong title={displayName}>{displayName}</strong>
                    <span className={styles.clientId} title="ID client"> #{client.id}</span>
                  </div>
                </td>
                <td>
                  <div className={styles.contactCell} title={contact.value !== '\u2014' ? contact.value : undefined}>
                    {contact.type === 'phone' && <FiPhone size={12} className={styles.contactIcon} />}
                    {contact.type === 'email' && <FiMail size={12} className={styles.contactIcon} />}
                    <span>{contact.value}</span>
                  </div>
                </td>
                <td className={styles.colSecondary}>
                  <div className={styles.addressCell} title={address.full || undefined}>
                    {address.short}
                  </div>
                </td>
                <td>
                  <span
                    className={`${styles.statusBadge} ${
                      client.is_active ? styles.statusActive : styles.statusInactive
                    }`}
                  >
                    {client.is_active ? 'Actif' : 'Inactif'}
                  </span>
                </td>
                <td className={`${styles.dateCell} ${styles.colSecondary}`}>
                  {formatDate(client.created_at)}
                </td>
                <td className={styles.tdActions} onClick={(e) => e.stopPropagation()}>
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
    </div>
  );
};

export default ClientsTable;
