import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FiMessageSquare, FiSend } from 'react-icons/fi';
import { format } from 'date-fns';
import { fetchBookingMessages as defaultFetchMessages, sendBookingMessage as defaultSendMessage } from '../../../../services/companyService';
import s from './BookingChat.module.css';

const SCROLL_THRESHOLD = 50;

/**
 * Mini-canal de communication booking.
 * Réutilisable côté company ET institution via les props:
 * - socket: instance socket.io (requis pour temps réel)
 * - fetchMessages: fn(bookingId, opts) (défaut: companyService.fetchBookingMessages)
 * - sendMessage: fn(bookingId, content) (défaut: companyService.sendBookingMessage)
 */
export default function BookingChat({
  bookingId,
  socket = null,
  fetchMessages: fetchMessagesProp = null,
  sendMessage: sendMessageProp = null,
  closed = false,
  /** Si true, affiche un court message au lieu de ne rien rendre quand l’API renvoie 404. */
  showUnavailableHint = false,
}) {
  const [messages, setMessages] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [text, setText] = useState('');
  const [unavailable, setUnavailable] = useState(false);
  // booking_id canonique du fil (A/R + multi-étapes unifiés côté backend).
  const [threadBookingId, setThreadBookingId] = useState(bookingId);
  const messagesEndRef = useRef(null);
  const messagesWrapRef = useRef(null);
  const debounceRef = useRef(null);
  const fetchMessagesApi = fetchMessagesProp || defaultFetchMessages;
  const sendMessageApi = sendMessageProp || defaultSendMessage;

  // Track if user is near bottom for autoscroll
  const isNearBottom = useCallback(() => {
    const wrap = messagesWrapRef.current;
    if (!wrap) return true;
    return wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight < SCROLL_THRESHOLD;
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Fetch initial messages
  useEffect(() => {
    if (!bookingId) return;
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      try {
        const data = await fetchMessagesApi(bookingId);
        if (!cancelled) {
          setMessages(data.messages || []);
          setHasMore(data.has_more || false);
          if (data.booking_id) setThreadBookingId(data.booking_id);
          setTimeout(scrollToBottom, 50);
        }
      } catch (err) {
        if (err?.response?.status === 404) {
          setUnavailable(true);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => { cancelled = true; };
  }, [bookingId, scrollToBottom, fetchMessagesApi]);

  // Socket listener for real-time messages
  useEffect(() => {
    if (!socket || !bookingId) return;

    const handler = (payload) => {
      // Accepte l'id du leg demandé OU le booking_id canonique du fil unifié.
      const incomingBookingId = payload?.booking_id;
      if (incomingBookingId !== bookingId && incomingBookingId !== threadBookingId) {
        return;
      }
      const incoming = payload.message;
      if (!incoming?.id) return;

      setMessages((prev) => {
        if (prev.some((m) => m.id === incoming.id)) return prev;
        return [...prev, incoming];
      });

      if (isNearBottom()) {
        setTimeout(scrollToBottom, 30);
      }
    };

    socket.on('booking_message', handler);
    return () => { socket.off('booking_message', handler); };
  }, [socket, bookingId, threadBookingId, isNearBottom, scrollToBottom]);

  // Send message with debounce
  const handleSend = useCallback(async () => {
    const content = text.trim();
    if (!content || sending) return;

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setSending(true);
      try {
        const data = await sendMessageApi(bookingId, content);
        if (data?.message) {
          setMessages((prev) => {
            if (prev.some((m) => m.id === data.message.id)) return prev;
            return [...prev, data.message];
          });
          setTimeout(scrollToBottom, 30);
        }
        setText('');
      } catch (err) {
        console.error('[BookingChat] send error', err);
      } finally {
        setSending(false);
      }
    }, 300);
  }, [text, sending, bookingId, scrollToBottom, sendMessageApi]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  // Load older messages
  const loadOlder = useCallback(async () => {
    if (!messages.length || !hasMore) return;
    const firstId = messages[0]?.id;
    try {
      const data = await fetchMessagesApi(bookingId, { beforeId: firstId });
      setMessages((prev) => [...(data.messages || []), ...prev]);
      setHasMore(data.has_more || false);
    } catch (err) {
      console.error('[BookingChat] load older error', err);
    }
  }, [messages, hasMore, bookingId, fetchMessagesApi]);

  // Format timestamp HH:mm
  const fmtTime = (iso) => {
    if (!iso) return '';
    try {
      return format(new Date(iso), 'HH:mm');
    } catch {
      return '';
    }
  };

  if (unavailable) {
    if (!showUnavailableHint) return null;
    return (
      <div className={s.chatSection}>
        <p className={s.emptyState}>
          Messagerie indisponible pour cette réservation (aucun canal ouvert).
        </p>
      </div>
    );
  }

  return (
    <div className={s.chatSection}>
      <div className={s.chatHeader}>
        <div className={s.chatIcon}><FiMessageSquare size={12} /></div>
        <h3 className={s.chatTitle}>Communication</h3>
      </div>

      {loading ? (
        <div className={s.loading}>Chargement...</div>
      ) : (
        <>
          <div className={s.messagesWrap} ref={messagesWrapRef}>
            {hasMore && (
              <div className={s.loadMore}>
                <button className={s.loadMoreBtn} onClick={loadOlder} type="button">
                  Charger les anciens messages
                </button>
              </div>
            )}

            {messages.length === 0 && (
              <div className={s.emptyState}>Aucun message</div>
            )}

            {messages.map((m) => {
              const isCompany = m.sender_type === 'COMPANY';
              const isClient = m.sender_type === 'CLIENT';
              return (
                <div
                  key={m.id}
                  className={`${s.messageRow} ${
                    isCompany ? s.messageRowCompany : isClient ? s.messageRowClient : s.messageRowInstitution
                  }`}
                >
                  <span className={s.senderLabel}>{m.sender_label}</span>
                  <div
                    className={`${s.bubble} ${
                      isCompany ? s.bubbleCompany : isClient ? s.bubbleClient : s.bubbleInstitution
                    }`}
                  >
                    {m.content}
                  </div>
                  <span className={s.timestamp}>{fmtTime(m.created_at)}</span>
                </div>
              );
            })}

            <div ref={messagesEndRef} />
          </div>

          {closed ? (
            <div className={s.closedBar}>Transport terminé — conversation clôturée</div>
          ) : (
            <div className={s.inputBar}>
              <input
                className={s.chatInput}
                type="text"
                placeholder="Ecrire un message..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={handleKeyDown}
                maxLength={500}
                disabled={sending}
              />
              <button
                className={s.sendBtn}
                onClick={handleSend}
                disabled={!text.trim() || sending}
                type="button"
                title="Envoyer"
              >
                <FiSend size={13} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
