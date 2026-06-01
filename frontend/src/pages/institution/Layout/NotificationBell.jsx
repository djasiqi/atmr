// pages/institution/Layout/NotificationBell.jsx
/**
 * Cloche de notifications in-app pour le header institution.
 *
 * - Badge compteur de non-lues
 * - Dropdown avec liste scrollable
 * - Clic sur une notif = marquer comme lue + navigation
 * - "Tout marquer comme lu"
 * - Temps réel via useInstitutionSocket (InstitutionLayout) : merge React Query local
 * - Polling React Query comme filet de sécurité (60s)
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  FaBell,
  FaPaperPlane,
  FaCheck,
  FaExchangeAlt,
  FaTruck,
  FaTimesCircle,
  FaBan,
  FaCommentDots,
} from 'react-icons/fa';
import {
  useInstitutionNotifications,
  useMarkNotificationRead,
  useMarkAllNotificationsRead,
} from '../../../hooks/useInstitutionData';
import { getAuthEnv } from '../../../utils/webAuthSession';
import styles from './NotificationBell.module.css';

// Icônes par type d'événement
const EVENT_ICONS = {
  request_sent: FaPaperPlane,
  offer_accepted: FaCheck,
  request_converted: FaExchangeAlt,
  booking_status_updated: FaTruck,
  request_cancelled: FaTimesCircle,
  booking_cancelled: FaBan,
  booking_message: FaCommentDots,
};

const EVENT_COLORS = {
  request_sent: '#3b82f6',
  offer_accepted: '#059669',
  request_converted: '#7c3aed',
  booking_status_updated: '#0891b2',
  request_cancelled: '#ef4444',
  booking_cancelled: '#dc2626',
  booking_message: '#0d9488',
};

/**
 * Calcule un horodatage relatif ("il y a 5 min", "il y a 2h", etc.)
 */
function timeAgo(dateString) {
  if (!dateString) return '';
  const now = new Date();
  const date = new Date(dateString);
  const diffMs = now - date;
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return "À l'instant";
  if (diffMin < 60) return `Il y a ${diffMin} min`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH < 24) return `Il y a ${diffH}h`;
  const diffD = Math.floor(diffH / 24);
  if (diffD < 7) return `Il y a ${diffD}j`;
  return date.toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit' });
}

const NotificationBell = () => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const isDemoEnv = getAuthEnv() === 'demo';
  const dashboardRoot = isDemoEnv ? '/demo/dashboard' : '/dashboard';
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);
  const bellRef = useRef(null);

  const { data, isLoading } = useInstitutionNotifications();
  const markRead = useMarkNotificationRead();
  const markAllRead = useMarkAllNotificationsRead();

  const notifications = data?.notifications || [];
  const unreadCount = data?.unread_count || 0;

  // Fermer au clic en dehors
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target) &&
        bellRef.current &&
        !bellRef.current.contains(e.target)
      ) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // Navigation vers la page concernée
  const getNotificationLink = useCallback(
    (notif) => {
      const meta = notif.metadata || {};
      const base = `${dashboardRoot}/institution/${public_id}`;
      if (meta.request_id) {
        return `${base}/requests/${meta.request_id}`;
      }
      return `${base}/requests`;
    },
    [public_id, dashboardRoot]
  );

  const handleNotificationClick = useCallback(
    (notif) => {
      if (!notif.is_read) {
        markRead.mutate(notif.id);
      }
      const link = getNotificationLink(notif);
      setIsOpen(false);
      navigate(link);
    },
    [markRead, getNotificationLink, navigate]
  );

  const handleMarkAllRead = useCallback(() => {
    markAllRead.mutate();
  }, [markAllRead]);

  return (
    <div className={styles.bellContainer}>
      {/* Cloche */}
      <button
        ref={bellRef}
        className={`${styles.bellBtn} ${unreadCount > 0 ? styles.hasUnread : ''}`}
        onClick={() => setIsOpen((prev) => !prev)}
        aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} non lues)` : ''}`}
        title="Notifications"
      >
        <FaBell className={styles.bellIcon} />
        {unreadCount > 0 && (
          <span className={styles.badge}>{unreadCount > 99 ? '99+' : unreadCount}</span>
        )}
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div ref={dropdownRef} className={styles.dropdown}>
          {/* Header */}
          <div className={styles.dropdownHeader}>
            <span className={styles.dropdownTitle}>Notifications</span>
            {unreadCount > 0 && (
              <button
                className={styles.markAllBtn}
                onClick={handleMarkAllRead}
                disabled={markAllRead.isPending}
              >
                Tout marquer comme lu
              </button>
            )}
          </div>

          {/* Liste */}
          <div className={styles.notifList}>
            {isLoading && (
              <div className={styles.emptyState}>Chargement...</div>
            )}

            {!isLoading && notifications.length === 0 && (
              <div className={styles.emptyState}>
                <FaBell className={styles.emptyIcon} />
                <p>Aucune notification</p>
              </div>
            )}

            {!isLoading &&
              notifications.map((notif) => {
                const Icon = EVENT_ICONS[notif.event_type] || FaBell;
                const color = EVENT_COLORS[notif.event_type] || '#64748b';

                return (
                  <button
                    key={notif.id}
                    className={`${styles.notifItem} ${!notif.is_read ? styles.unread : ''}`}
                    onClick={() => handleNotificationClick(notif)}
                  >
                    <div
                      className={styles.notifIcon}
                      style={{ backgroundColor: `${color}15`, color }}
                    >
                      <Icon />
                    </div>
                    <div className={styles.notifContent}>
                      <div className={styles.notifTitle}>{notif.title}</div>
                      <div className={styles.notifMessage}>{notif.message}</div>
                      <div className={styles.notifTime}>{timeAgo(notif.created_at)}</div>
                    </div>
                    {!notif.is_read && <div className={styles.unreadDot} />}
                  </button>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
};

export default NotificationBell;
