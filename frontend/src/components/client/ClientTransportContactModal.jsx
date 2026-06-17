import React, { useEffect, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import BookingChat from '../../pages/company/Reservations/components/BookingChat';
import { ensureClientPortalSocket } from '../../services/clientPortalSocket';
import { fetchBookingMessagesForClient, sendBookingMessageAsClient } from '../../services/bookingService';
import { isBookingChatClosed } from '../../utils/bookingChat';
import styles from '../../pages/client/Reservations/Reservations.module.css';

/**
 * Modal « Contacter le transporteur » : numéro affiché + mini-chat (si entreprise retenue).
 */
export default function ClientTransportContactModal({ booking, open, onClose, onOpenSupport }) {
  const [portalEl, setPortalEl] = useState(null);

  useEffect(() => {
    setPortalEl(typeof document !== 'undefined' ? document.body : null);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  if (!open || !booking || !portalEl) return null;

  const companyName = String(booking.company_name || 'Transporteur').trim();
  const rawPhone = String(booking.company_contact_phone || '').replace(/\s/g, '');
  const hasDialablePhone = rawPhone.length >= 8 && /^\+?\d/.test(rawPhone);
  const hasCompany = Boolean(booking.company_id);
  const chatClosed = isBookingChatClosed(booking);
  const showChat = hasCompany && !chatClosed;

  return createPortal(
    <div className={styles.contactModalBackdrop} role="presentation" onClick={onClose}>
      <div
        className={styles.contactModal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="client-contact-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.contactModalHeader}>
          <h2 id="client-contact-modal-title" className={styles.contactModalTitle}>
            Contacter {companyName}
          </h2>
          <button type="button" className={styles.contactModalClose} onClick={onClose} aria-label="Fermer">
            ×
          </button>
        </div>
        <div className={styles.contactModalBody}>
          {hasDialablePhone ? (
            <div className={styles.contactModalPhoneBlock}>
              <p className={styles.contactModalPhoneLabel}>Téléphone du transporteur</p>
              <a className={styles.contactModalPhoneLink} href={`tel:${rawPhone}`}>
                {String(booking.company_contact_phone || '').trim() || rawPhone}
              </a>
            </div>
          ) : (
            <p className={styles.contactModalMuted}>
              {hasCompany
                ? 'Numéro du transporteur non renseigné ici. Utilisez le fil de discussion ci-dessous ou les coordonnées reçues par courriel ou SMS.'
                : 'Votre demande n’est pas encore attribuée à une entreprise : le numéro sera affiché dès qu’un transporteur acceptera la course.'}
            </p>
          )}

          {showChat ? (
            <div className={styles.contactModalChatSection}>
              <p className={styles.contactModalChatIntro}>Échange écrit lié à cette course (comme avec une institution)</p>
              <ClientBookingChatBridge bookingId={booking.id} closed={chatClosed} />
            </div>
          ) : hasCompany && chatClosed ? (
            <p className={styles.contactModalMuted}>Cette course est terminée ou annulée : la messagerie est fermée.</p>
          ) : null}

          <div className={styles.contactModalSupportRow}>
            <button type="button" className="secondaryButton" onClick={onOpenSupport}>
              Contacter le support LIRIE
            </button>
          </div>
        </div>
      </div>
    </div>,
    portalEl
  );
}

/**
 * Charge le socket portail client et branche BookingChat (fetch/send dédiés client).
 */
function ClientBookingChatBridge({ bookingId, closed }) {
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const s = await ensureClientPortalSocket();
      if (!cancelled) setSocket(s || null);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const fetchMessages = useCallback((id, opts) => fetchBookingMessagesForClient(id, opts), []);
  const sendMessage = useCallback((id, content) => sendBookingMessageAsClient(id, content), []);

  if (!socket) {
    return <p className={styles.contactModalMuted}>Connexion au fil de discussion…</p>;
  }

  return (
    <div className={styles.contactModalChatMount}>
      <BookingChat
        bookingId={bookingId}
        socket={socket}
        closed={closed}
        fetchMessages={fetchMessages}
        sendMessage={sendMessage}
        showUnavailableHint
      />
    </div>
  );
}
