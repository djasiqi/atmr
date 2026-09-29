import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiBell } from 'react-icons/fi';
import { fetchClientMessageNotifications } from '../../../services/clientService';
import { ensureClientPortalSocket } from '../../../services/clientPortalSocket';
import {
  CLIENT_MESSAGE_READ_EVENT,
  clientNotificationReadKey,
  readClientMessageIds,
  rememberClientMessageRead,
} from '../../../utils/clientMessageNotifRead';
import styles from './HeaderDashboard.module.css';

function timeAgo(dateString) {
  if (!dateString) return '';
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return '';
  const diffMin = Math.floor((Date.now() - date.getTime()) / 60000);
  if (diffMin < 1) return "À l'instant";
  if (diffMin < 60) return `Il y a ${diffMin} min`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH < 24) return `Il y a ${diffH} h`;
  return date.toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit' });
}

const ClientNotificationBell = ({ publicId }) => {
  const navigate = useNavigate();
  const wrapRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState([]);
  const [read, setRead] = useState(() => readClientMessageIds());

  const load = useCallback(async () => {
    try {
      const data = await fetchClientMessageNotifications();
      setItems(Array.isArray(data?.notifications) ? data.notifications : []);
    } catch (error) {
      console.error('[ClientNotificationBell]', error);
    }
  }, []);

  useEffect(() => {
    void load();
    const onFocus = () => {
      void load();
    };
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [load]);

  useEffect(() => {
    const syncRead = () => setRead(readClientMessageIds());
    window.addEventListener(CLIENT_MESSAGE_READ_EVENT, syncRead);
    return () => window.removeEventListener(CLIENT_MESSAGE_READ_EVENT, syncRead);
  }, []);

  useEffect(() => {
    let socket;
    let cancelled = false;
    const onMessage = (payload) => {
      const sender = String(payload?.message?.sender_type || '').toUpperCase();
      if (sender && sender !== 'COMPANY') return;
      void load();
    };
    ensureClientPortalSocket().then((connected) => {
      if (cancelled || !connected) return;
      socket = connected;
      socket.on('booking_message', onMessage);
    });
    return () => {
      cancelled = true;
      socket?.off('booking_message', onMessage);
    };
  }, [load]);

  useEffect(() => {
    if (!open) return undefined;
    const onPointer = (event) => {
      if (wrapRef.current && !wrapRef.current.contains(event.target)) setOpen(false);
    };
    const onKey = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const unread = items.filter((item) => !read.has(clientNotificationReadKey(item.id)));

  const openItem = (item) => {
    const next = rememberClientMessageRead([item.id]);
    setRead(next);
    setOpen(false);
    if (item.event_type === 'invoice_received') {
      if (publicId) navigate(`/factures/${publicId}`);
      return;
    }
    if (!publicId || !item.booking_id) return;
    navigate(`/reservations/${publicId}?contact=${item.booking_id}`);
  };

  const markAll = () => {
    const next = rememberClientMessageRead(items.map((item) => item.id));
    setRead(next);
  };

  return (
    <div className={styles.bellWrap} ref={wrapRef}>
      <button
        type="button"
        className={styles.bellButton}
        onClick={() => setOpen((value) => !value)}
        aria-label={unread.length ? `Notifications, ${unread.length} non lues` : 'Notifications'}
        aria-expanded={open}
      >
        <FiBell className={styles.bellIcon} aria-hidden />
        {unread.length > 0 ? (
          <span className={styles.bellBadge}>{unread.length > 9 ? '9+' : unread.length}</span>
        ) : null}
      </button>
      {open ? (
        <div className={styles.bellMenu} role="dialog" aria-label="Notifications">
          <div className={styles.bellMenuHeader}>
            <span>Notifications</span>
            {unread.length > 0 ? (
              <button type="button" className={styles.bellMenuMark} onClick={markAll}>
                Tout marquer comme lu
              </button>
            ) : null}
          </div>
          {items.length === 0 ? (
            <p className={styles.bellMenuEmpty}>Aucune notification pour le moment.</p>
          ) : (
            <ul className={styles.bellMenuList}>
              {items.map((item) => {
                const isUnread = !read.has(clientNotificationReadKey(item.id));
                return (
                  <li key={item.id}>
                    <button
                      type="button"
                      className={`${styles.bellMenuItem}${isUnread ? ` ${styles.bellMenuItemUnread}` : ''}`}
                      onClick={() => openItem(item)}
                    >
                      <span className={styles.bellMenuTitle}>{item.title}</span>
                      <span className={styles.bellMenuText}>{item.message}</span>
                      <span className={styles.bellMenuTime}>{timeAgo(item.created_at)}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      ) : null}
    </div>
  );
};

export default ClientNotificationBell;
