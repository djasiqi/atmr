// components/layout/Header/CompanyNotificationBell.jsx
/**
 * Cloche de notifications in-app pour le header entreprise.
 *
 * - Badge compteur non-lues
 * - Dropdown avec liste scrollable (cache TanStack Query, affichage instantané)
 * - Clic = navigation immédiate + marquage lu en arrière-plan
 * - "Tout marquer comme lu"
 * - Écoute socket new_company_notification pour temps réel
 */

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  FaBell,
  FaCommentDots,
  FaAmbulance,
  FaBan,
  FaExchangeAlt,
} from 'react-icons/fa';
import { resolveCompanyNotificationLink } from '../../../utils/companyNotificationNavigation';
import {
  fetchCompanyNotifications,
  markCompanyNotificationRead,
  markAllCompanyNotificationsRead,
} from '../../../services/companyService';
import useCompanySocket from '../../../hooks/useCompanySocket';
import { getAccessToken } from '../../../hooks/useAuthToken';
import { useLirieCompany } from '../../../hooks/useLirieCompany';
import { getCurrentAuthEnv } from '../../../utils/apiClient';
import { getActiveUser, hasCompanyScopedAccessToken } from '../../../utils/webAuthSession';
import { recordDashboardApiCall } from '../../../utils/companyDashboardDuplicationReport';
import { isCompanyDashboardPerfEnabled } from '../../../utils/companyDashboardPerfInstrumentation';
import { LIRIE_QK_PREFIX, lirieKeys } from '../../../queryKeys/lirie';
import styles from './CompanyNotificationBell.module.css';

/** Aligné sur useLirieCompany : cookies httpOnly ou token legacy authToken suffisent. */
const canLoadCompanyNotifications = () => {
  const env = getCurrentAuthEnv();
  return (
    hasCompanyScopedAccessToken(env) ||
    Boolean(getAccessToken()) ||
    Boolean(getActiveUser())
  );
};

const EVENT_ICONS = {
  booking_message: FaCommentDots,
  new_request: FaAmbulance,
  institution_change_request: FaExchangeAlt,
};

const EVENT_COLORS = {
  booking_message: '#0d9488',
  new_request: '#d97706',
  institution_change_request: '#0d9488',
};

const CANCEL_COLOR = '#ef4444';
const CHANGE_REQUEST_COLOR = '#0d9488';

/** Cache local : la cloche reste réactive entre ouvertures. */
const NOTIFICATIONS_STALE_MS = 60_000;
/** Resync HTTP périodique long (pas de GET sur chaque event socket). */
const NOTIFICATIONS_RESYNC_INTERVAL_MS = 10 * 60 * 1000;

function resolveNotificationVisual(notif) {
  const meta = notif?.metadata && typeof notif.metadata === 'object' ? notif.metadata : {};
  const actionType = String(meta.action_type || '').toUpperCase();
  const title = String(notif?.title || '');
  const isCancellation =
    actionType === 'CANCELLATION'
    || /annulation/i.test(title);

  if (notif?.event_type === 'institution_change_request') {
    if (isCancellation) {
      return { Icon: FaBan, color: CANCEL_COLOR };
    }
    return { Icon: FaExchangeAlt, color: CHANGE_REQUEST_COLOR };
  }

  return {
    Icon: EVENT_ICONS[notif?.event_type] || FaBell,
    color: EVENT_COLORS[notif?.event_type] || '#64748b',
  };
}

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
  const queryClient = useQueryClient();
  const isDemoEnv = (getCurrentAuthEnv() || '').toLowerCase() === 'demo';
  const dashboardRoot = isDemoEnv ? '/demo/dashboard' : '/dashboard';
  const socket = useCompanySocket();
  const { company } = useLirieCompany();

  const [isOpen, setIsOpen] = useState(false);
  const [markingAllRead, setMarkingAllRead] = useState(false);
  const dropdownRef = useRef(null);
  const bellRef = useRef(null);

  const notificationsQueryKey = useMemo(
    () => (company?.id ? lirieKeys.companyNotifications(company.id) : null),
    [company?.id]
  );

  const canLoad = canLoadCompanyNotifications();

  const {
    data: inboxData,
    isLoading,
    isFetching,
  } = useQuery({
    queryKey: notificationsQueryKey ?? [LIRIE_QK_PREFIX, 'company-notifications', 'disabled'],
    queryFn: async () => {
      if (isCompanyDashboardPerfEnabled()) {
        recordDashboardApiCall({
          key: 'alerts',
          url: '/companies/notifications',
          componentId: 'CompanyNotificationBell',
          callerStack: new Error().stack,
        });
      }
      return fetchCompanyNotifications({ limit: 30 });
    },
    enabled: Boolean(notificationsQueryKey && canLoad),
    staleTime: NOTIFICATIONS_STALE_MS,
    refetchInterval: NOTIFICATIONS_RESYNC_INTERVAL_MS,
    refetchIntervalInBackground: false,
  });

  const notifications = inboxData?.notifications ?? [];
  const unreadCount = inboxData?.unread_count ?? 0;
  const showInitialLoader = isLoading && notifications.length === 0;
  const isBackgroundRefresh = isFetching && notifications.length > 0;

  const prefetchNotifications = useCallback(() => {
    if (!notificationsQueryKey || !canLoad) return;
    void queryClient.prefetchQuery({
      queryKey: notificationsQueryKey,
      queryFn: () => fetchCompanyNotifications({ limit: 30 }),
      staleTime: NOTIFICATIONS_STALE_MS,
    });
  }, [canLoad, notificationsQueryKey, queryClient]);

  const mergeIncomingNotification = useCallback(
    (raw) => {
      if (!raw?.id || !notificationsQueryKey) return;
      queryClient.setQueryData(notificationsQueryKey, (prev) => {
        if (!prev) return prev;
        if ((prev.notifications || []).some((n) => n.id === raw.id)) return prev;
        const incoming = {
          ...raw,
          metadata: raw.metadata != null ? raw.metadata : {},
        };
        return {
          ...prev,
          notifications: [incoming, ...(prev.notifications || [])].slice(0, 30),
          unread_count: (prev.unread_count || 0) + (incoming.is_read ? 0 : 1),
        };
      });
    },
    [notificationsQueryKey, queryClient]
  );

  useEffect(() => {
    const onAuthChanged = () => {
      setIsOpen(false);
      void queryClient.invalidateQueries({
        queryKey: [LIRIE_QK_PREFIX, 'company-notifications'],
      });
    };
    window.addEventListener('auth-changed', onAuthChanged);
    return () => window.removeEventListener('auth-changed', onAuthChanged);
  }, [queryClient]);

  useEffect(() => {
    if (!socket || !notificationsQueryKey) return;

    const onNewNotification = (payload) => {
      mergeIncomingNotification(payload);
    };

    const onOfferUpdated = (payload) => {
      if (payload?.is_relaunch) {
        void queryClient.invalidateQueries({ queryKey: notificationsQueryKey });
      }
    };

    socket.on('new_company_notification', onNewNotification);
    socket.on('institution_offer_updated', onOfferUpdated);
    return () => {
      socket.off('new_company_notification', onNewNotification);
      socket.off('institution_offer_updated', onOfferUpdated);
    };
  }, [socket, notificationsQueryKey, mergeIncomingNotification, queryClient]);

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
    (notif) => {
      const link = resolveCompanyNotificationLink({
        notif,
        dashboardRoot,
        companyPublicId: public_id,
      });

      setIsOpen(false);
      navigate(link);

      if (!notif.is_read && notificationsQueryKey) {
        queryClient.setQueryData(notificationsQueryKey, (prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            notifications: (prev.notifications || []).map((n) =>
              n.id === notif.id ? { ...n, is_read: true } : n
            ),
            unread_count: Math.max(0, (prev.unread_count || 0) - 1),
          };
        });
        void markCompanyNotificationRead(notif.id).catch((err) => {
          console.error('[CompanyNotificationBell] Mark read error:', err);
          void queryClient.invalidateQueries({ queryKey: notificationsQueryKey });
        });
      }
    },
    [public_id, navigate, dashboardRoot, notificationsQueryKey, queryClient]
  );

  const handleMarkAllRead = useCallback(async () => {
    if (!notificationsQueryKey) return;
    try {
      setMarkingAllRead(true);
      queryClient.setQueryData(notificationsQueryKey, (prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          notifications: (prev.notifications || []).map((n) => ({ ...n, is_read: true })),
          unread_count: 0,
        };
      });
      await markAllCompanyNotificationsRead();
    } catch (err) {
      console.error('[CompanyNotificationBell] Mark all read error:', err);
      void queryClient.invalidateQueries({ queryKey: notificationsQueryKey });
    } finally {
      setMarkingAllRead(false);
    }
  }, [notificationsQueryKey, queryClient]);

  const handleBellClick = useCallback(() => {
    setIsOpen((prev) => !prev);
  }, []);

  return (
    <div className={styles.bellContainer}>
      <button
        ref={bellRef}
        className={`${styles.bellBtn} ${unreadCount > 0 ? styles.hasUnread : ''}`}
        onClick={handleBellClick}
        onMouseEnter={prefetchNotifications}
        onFocus={prefetchNotifications}
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
            <span className={styles.dropdownTitle}>
              Notifications
              {isBackgroundRefresh && (
                <span className={styles.refreshHint} aria-hidden="true" />
              )}
            </span>
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
            {showInitialLoader && (
              <div className={styles.emptyState}>Chargement...</div>
            )}

            {!showInitialLoader && notifications.length === 0 && (
              <div className={styles.emptyState}>
                <FaBell className={styles.emptyIcon} />
                <p>Aucune notification</p>
              </div>
            )}

            {notifications.map((notif) => {
              const { Icon, color } = resolveNotificationVisual(notif);

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
