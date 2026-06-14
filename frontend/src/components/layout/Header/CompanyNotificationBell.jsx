// components/layout/Header/CompanyNotificationBell.jsx
/**
 * Cloche de notifications in-app pour le header entreprise.
 *
 * - Badge compteur non-lues
 * - Dropdown avec liste scrollable
 * - Clic = marquer comme lue + navigation vers la réservation
 * - "Tout marquer comme lu"
 * - Écoute socket new_company_notification pour temps réel
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  FaBell,
  FaCommentDots,
  FaAmbulance,
  FaEdit,
  FaExclamationTriangle,
} from 'react-icons/fa';
import { resolveCompanyNotificationLink } from '../../../utils/companyNotificationNavigation';
import {
  fetchCompanyNotifications,
  markCompanyNotificationRead,
  markAllCompanyNotificationsRead,
} from '../../../services/companyService';
import useCompanySocket from '../../../hooks/useCompanySocket';
import { getCurrentAuthEnv } from '../../../utils/apiClient';
import { hasCompanyScopedAccessToken } from '../../../utils/webAuthSession';
import { recordDashboardApiCall } from '../../../utils/companyDashboardDuplicationReport';
import { isCompanyDashboardPerfEnabled } from '../../../utils/companyDashboardPerfInstrumentation';
import styles from './CompanyNotificationBell.module.css';

const hasCompanyToken = () => hasCompanyScopedAccessToken(getCurrentAuthEnv());

const EVENT_ICONS = {
  booking_message: FaCommentDots,
  new_request: FaAmbulance,
};

const EVENT_COLORS = {
  booking_message: '#0d9488',
  new_request: '#d97706',
};

/** Resync HTTP périodique long (pas de GET sur chaque event socket). */
const NOTIFICATIONS_RESYNC_INTERVAL_MS = 10 * 60 * 1000;

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

const CompanyNotificationBell = () => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const isDemoEnv = (getCurrentAuthEnv() || '').toLowerCase() === 'demo';
  const dashboardRoot = isDemoEnv ? '/demo/dashboard' : '/dashboard';
  const socket = useCompanySocket();

  const [isOpen, setIsOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [markingAllRead, setMarkingAllRead] = useState(false);
  const dropdownRef = useRef(null);
  const bellRef = useRef(null);
  /** IDs déjà présents (GET + socket) — dédup stricte. */
  const seenNotificationIdsRef = useRef(new Set());
  /** Max created_at (ms) des entrées fusionnées (stats / cohérence). */
  const lastSeenNotificationTsRef = useRef(0);

  const loadNotifications = useCallback(async () => {
    if (!hasCompanyToken()) {
      setNotifications([]);
      setUnreadCount(0);
      seenNotificationIdsRef.current = new Set();
      lastSeenNotificationTsRef.current = 0;
      return;
    }
    try {
      setIsLoading(true);
      if (isCompanyDashboardPerfEnabled()) {
        recordDashboardApiCall({
          key: 'alerts',
          url: '/companies/notifications',
          componentId: 'CompanyNotificationBell',
          callerStack: new Error().stack,
        });
      }
      const data = await fetchCompanyNotifications({ limit: 30 });
      const list = data.notifications || [];
      setNotifications(list);
      setUnreadCount(data.unread_count || 0);
      const seen = new Set();
      let maxTs = 0;
      for (const n of list) {
        if (n.id != null) seen.add(n.id);
        if (n.created_at) {
          const t = new Date(n.created_at).getTime();
          if (Number.isFinite(t)) maxTs = Math.max(maxTs, t);
        }
      }
      seenNotificationIdsRef.current = seen;
      lastSeenNotificationTsRef.current = maxTs;
    } catch (err) {
      console.error('[CompanyNotificationBell] Load error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const mergeIncomingNotification = useCallback((raw) => {
    if (!raw || raw.id == null) return;
    const id = raw.id;
    if (seenNotificationIdsRef.current.has(id)) return;

    const createdMs = raw.created_at ? new Date(raw.created_at).getTime() : Date.now();
    if (!Number.isFinite(createdMs)) return;

    seenNotificationIdsRef.current.add(id);
    lastSeenNotificationTsRef.current = Math.max(lastSeenNotificationTsRef.current, createdMs);

    const incoming = {
      ...raw,
      metadata: raw.metadata != null ? raw.metadata : {},
    };

    setNotifications((prev) => {
      if (prev.some((n) => n.id === id)) return prev;
      return [incoming, ...prev].slice(0, 30);
    });
    if (!incoming.is_read) {
      setUnreadCount((c) => c + 1);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadNotifications();
  }, [loadNotifications]);

  // Resync long (pas de refetch à chaque event socket)
  useEffect(() => {
    const interval = setInterval(loadNotifications, NOTIFICATIONS_RESYNC_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [loadNotifications]);

  // Socket temps réel — merge local sans GET
  useEffect(() => {
    if (!socket) return;

    const handler = (payload) => {
      mergeIncomingNotification(payload);
    };

    socket.on('new_company_notification', handler);
    return () => {
      socket.off('new_company_notification', handler);
    };
  }, [socket, mergeIncomingNotification]);

  // Close on click outside
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

  const handleNotificationClick = useCallback(
    async (notif) => {
      if (!notif.is_read) {
        try {
          await markCompanyNotificationRead(notif.id);
          setNotifications((prev) =>
            prev.map((n) => (n.id === notif.id ? { ...n, is_read: true } : n))
          );
          setUnreadCount((c) => Math.max(0, c - 1));
        } catch (err) {
          console.error('[CompanyNotificationBell] Mark read error:', err);
        }
      }

      const link = resolveCompanyNotificationLink({
        notif,
        dashboardRoot,
        companyPublicId: public_id,
      });

      setIsOpen(false);
      navigate(link);
    },
    [public_id, navigate, dashboardRoot]
  );

  const handleMarkAllRead = useCallback(async () => {
    try {
      setMarkingAllRead(true);
      await markAllCompanyNotificationsRead();
      setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
      setUnreadCount(0);
    } catch (err) {
      console.error('[CompanyNotificationBell] Mark all read error:', err);
    } finally {
      setMarkingAllRead(false);
    }
  }, []);

  return (
    <div className={styles.bellContainer}>
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

      {isOpen && (
        <div ref={dropdownRef} className={styles.dropdown}>
          <div className={styles.dropdownHeader}>
            <span className={styles.dropdownTitle}>Notifications</span>
            {unreadCount > 0 && (
              <button
                className={styles.markAllBtn}
                onClick={handleMarkAllRead}
                disabled={markingAllRead}
              >
                Tout marquer comme lu
              </button>
            )}
          </div>

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

export default CompanyNotificationBell;
