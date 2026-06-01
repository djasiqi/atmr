// src/pages/company/components/CompanyDriverTable.jsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { FiEdit, FiTrash2, FiMoreVertical, FiEye, FiPower, FiAlertTriangle, FiPhone } from 'react-icons/fi';
import s from './CompanyDriverTable.module.css';
import { formatLastSeen, getFreshnessStatus } from '../../../utils/mapUtils';
import {
  isDriverConstrained,
  getDriverConstraintReason,
} from '../../../utils/companyDriverProjections';

const CompanyDriverTable = ({ drivers, onEdit, onToggleStatus, onDeleteRequest }) => {
  const [openMenu, setOpenMenu] = useState(null);
  const menuRef = useRef(null);

  const closeMenu = useCallback(() => setOpenMenu(null), []);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        closeMenu();
      }
    };
    if (openMenu !== null) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [openMenu, closeMenu]);

  const getDisplayName = (driver) => {
    if (driver.first_name || driver.last_name) {
      return `${driver.first_name || ''} ${driver.last_name || ''}`.trim();
    }
    return driver.full_name || driver.username || driver.name || `Chauffeur #${driver.id}`;
  };

  const getAvailabilityStatus = (driver) => {
    if (!driver.is_active) return { label: 'Hors ligne', className: s.statusOffline };
    const backendStatus = String(driver?.status || '').toLowerCase();
    if (backendStatus === 'busy') return { label: 'En course', className: s.statusOnTrip };
    if (backendStatus === 'assigned') return { label: 'Assigné', className: s.statusOnTrip };
    if (backendStatus === 'offline') return { label: 'Hors ligne', className: s.statusOffline };
    return { label: 'Disponible', className: s.statusAvailable };
  };

  const getFreshnessLabel = (driver) => {
    const status = getFreshnessStatus(driver);
    if (status === 'live') return `Live · ${formatLastSeen(driver.last_seen_seconds)}`;
    if (status === 'recent') return `Recent · ${formatLastSeen(driver.last_seen_seconds)}`;
    if (status === 'stale') return `Stale · ${formatLastSeen(driver.last_seen_seconds)}`;
    return 'Offline';
  };

  return (
    <div className={s.tableContainer}>
      <table className={s.table}>
        <thead>
          <tr>
            <th>Chauffeur</th>
            <th>Vehicule</th>
            <th>Disponibilite</th>
            <th>Fraicheur</th>
            <th>Compte</th>
            <th className={s.thActions}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {(drivers || []).map((driver) => {
            const displayName = getDisplayName(driver);
            const availability = getAvailabilityStatus(driver);
            const initials = displayName
              .split(' ')
              .slice(0, 2)
              .map((w) => w[0]?.toUpperCase() || '')
              .join('');

            return (
              <tr key={driver.id} onClick={() => onEdit(driver)} className={s.row}>
                <td>
                  <div className={s.driverCell}>
                    {driver.photo ? (
                      <img
                        src={driver.photo}
                        alt=""
                        className={s.avatar}
                      />
                    ) : (
                      <div className={s.avatarFallback}>{initials || 'CH'}</div>
                    )}
                    <div className={s.driverInfo}>
                      <span className={s.driverName}>{displayName}</span>
                      {driver.email && (
                        <span className={s.driverSub}>{driver.email}</span>
                      )}
                    </div>
                  </div>
                </td>
                <td>
                  <span className={s.vehicleText}>
                    {driver.vehicle_assigned || '\u2014'}
                  </span>
                </td>
                <td>
                  <div className={s.statusCell}>
                    <span className={`${s.statusBadge} ${availability.className}`}>
                      {availability.label}
                    </span>
                    {isDriverConstrained(driver) && (
                      <span
                        className={`${s.statusBadge} ${s.statusConstrained}`}
                        title={`Batterie restreinte — l'app du chauffeur signale un problème d'optimisation batterie (raison : ${
                          getDriverConstraintReason(driver) || 'inconnue'
                        }). Position figée.`}
                        role="status"
                      >
                        <FiAlertTriangle size={11} aria-hidden />
                        <span>Batterie restreinte</span>
                      </span>
                    )}
                    {isDriverConstrained(driver) && driver.phone ? (
                      <a
                        href={`tel:${driver.phone}`}
                        className={s.contactLink}
                        onClick={(e) => e.stopPropagation()}
                        title={`Contacter le chauffeur (${driver.phone})`}
                      >
                        <FiPhone size={11} aria-hidden />
                        <span>Contacter le chauffeur</span>
                      </a>
                    ) : null}
                  </div>
                </td>
                <td>
                  <span className={s.driverSub}>
                    {getFreshnessLabel(driver)}
                  </span>
                </td>
                <td>
                  <span className={`${s.accountBadge} ${driver.is_active ? s.accountActive : s.accountInactive}`}>
                    {driver.is_active ? 'Actif' : 'Inactif'}
                  </span>
                </td>
                <td className={s.actionsCell} onClick={(e) => e.stopPropagation()}>
                  <button
                    type="button"
                    className={s.editBtn}
                    onClick={() => onEdit(driver)}
                    title="Modifier"
                  >
                    <FiEdit size={14} />
                  </button>
                  <div className={s.menuWrap} ref={openMenu === driver.id ? menuRef : null}>
                    <button
                      type="button"
                      className={s.menuTrigger}
                      onClick={() => setOpenMenu(openMenu === driver.id ? null : driver.id)}
                      title="Plus d'actions"
                    >
                      <FiMoreVertical size={14} />
                    </button>
                    {openMenu === driver.id && (
                      <div className={s.menuDropdown}>
                        <button
                          type="button"
                          className={s.menuItem}
                          onClick={() => {
                            onEdit(driver);
                            closeMenu();
                          }}
                        >
                          <FiEye size={13} />
                          Voir la fiche
                        </button>
                        <button
                          type="button"
                          className={s.menuItem}
                          onClick={() => {
                            onToggleStatus(driver.id, !driver.is_active);
                            closeMenu();
                          }}
                        >
                          <FiPower size={13} />
                          {driver.is_active ? 'Desactiver' : 'Activer'}
                        </button>
                        <button
                          type="button"
                          className={`${s.menuItem} ${s.menuItemDanger}`}
                          onClick={() => {
                            onDeleteRequest(driver);
                            closeMenu();
                          }}
                        >
                          <FiTrash2 size={13} />
                          Supprimer
                        </button>
                      </div>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default CompanyDriverTable;
