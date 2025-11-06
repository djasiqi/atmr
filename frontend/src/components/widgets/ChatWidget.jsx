// src/components/widgets/ChatWidget.jsx
import React, { useState, useEffect, useRef } from 'react';
import { FiMessageSquare, FiX } from 'react-icons/fi';
import useCompanySocket from '../../hooks/useCompanySocket';
import apiClient from '../../utils/apiClient';
import { v4 as uuidv4 } from 'uuid';
import './ChatWidget.css';

export default function ChatWidget({ companyId }) {
  const socket = useCompanySocket();
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [hasMore, setHasMore] = useState(true);
  const [oldestTs, setOldestTs] = useState(null);
  const listRef = useRef();
  const localIdsRef = useRef(new Set());

  const [myName] = useState(() => {
    try {
      const user = JSON.parse(localStorage.getItem('user') || '{}');
      return user.username || 'Moi';
    } catch {
      return 'Moi';
    }
  });

  // Helper pour formater la date
  const formatDate = (iso) => {
    const d = new Date(iso);
    const day = String(d.getDate()).padStart(2, '0');
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const year = d.getFullYear();
    return `${day}.${month}.${year}`;
  };

  // 1️⃣ Chargement initial (20 derniers messages)
  useEffect(() => {
    if (!open || !companyId) return;
    apiClient
      .get(`/messages/${companyId}`, { params: { limit: 20 } })
      .then((res) => {
        const raw = Array.isArray(res.data.messages) ? res.data.messages : res.data;
        const formatted = raw.map((m) => ({
          id: m.id,
          company_id: companyId,
          content: m.content,
          timestamp: m.timestamp,
          sender_name: m.sender_name || m.sender,
          receiver_name: m.receiver_name || m.receiver,
          sender_role: m.sender_role,
          _localId: null, // pas d'écho côté GET
        }));
        setMessages(formatted);
        if (formatted.length < 20) setHasMore(false);
        if (formatted.length > 0) setOldestTs(formatted[0].timestamp);
      })
      .catch((err) => console.error('❌ Erreur loading messages:', err));
  }, [open, companyId]);

  // 2️⃣ Scroll en bas après le premier chargement
  useEffect(() => {
    if (messages.length && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages]);

  // 3️⃣ Infinite scroll ascendant pour loader les anciens
  useEffect(() => {
    const el = listRef.current;
    if (!el || !hasMore) return;

    const onScroll = () => {
      if (el.scrollTop === 0 && hasMore) {
        const prevHeight = el.scrollHeight;
        apiClient
          .get(`/messages/${companyId}`, {
            params: { limit: 20, before: oldestTs },
          })
          .then((res) => {
            const raw = Array.isArray(res.data.messages) ? res.data.messages : res.data;
            const formatted = raw.map((m) => ({
              id: m.id,
              company_id: companyId,
              content: m.content,
              timestamp: m.timestamp,
              sender_name: m.sender_name || m.sender,
              receiver_name: m.receiver_name || m.receiver,
              sender_role: m.sender_role,
              _localId: null,
            }));
            setMessages((prev) => [...formatted, ...prev]);
            if (formatted.length < 20) setHasMore(false);
            if (formatted.length > 0) setOldestTs(formatted[0].timestamp);
            setTimeout(() => {
              el.scrollTop = el.scrollHeight - prevHeight;
            }, 0);
          })
          .catch(console.error);
      }
    };

    el.addEventListener('scroll', onScroll);
    return () => el.removeEventListener('scroll', onScroll);
  }, [companyId, oldestTs, hasMore]);

  // 4️⃣ Réception temps réel via WebSocket
  useEffect(() => {
    if (!socket || !companyId) {
      console.warn('[ChatWidget] ⚠️ Socket ou companyId manquant:', {
        socket: !!socket,
        companyId,
      });
      return;
    }

    console.log('[ChatWidget] 🔌 Écoute des messages WebSocket pour companyId=', companyId);

    const handleMessage = (msg) => {
      console.log('[ChatWidget] 📨 Message reçu via WebSocket:', msg);
      if (msg._localId && localIdsRef.current.has(msg._localId)) {
        console.log('[ChatWidget] 🔄 Message local détecté, suppression du set:', msg._localId);
        localIdsRef.current.delete(msg._localId);
        return;
      }
      if (msg.company_id !== companyId) {
        console.log(
          '[ChatWidget] ⚠️ Message ignoré (company_id différent):',
          msg.company_id,
          'vs',
          companyId
        );
        return;
      }
      console.log('[ChatWidget] ✅ Ajout du message à la liste');
      setMessages((prev) => [...prev, msg]);
    };

    const handleError = (error) => {
      console.error('[ChatWidget] ❌ Erreur WebSocket:', error);
    };

    socket.on('team_chat_message', handleMessage);
    socket.on('error', handleError);

    return () => {
      socket.off('team_chat_message', handleMessage);
      socket.off('error', handleError);
    };
  }, [socket, companyId]);

  // 5️⃣ Envoi d'un message (optimistic update)
  const sendMessage = () => {
    const text = input.trim();
    if (!text || !socket || !companyId) {
      console.error(
        "[ChatWidget] ❌ Impossible d'envoyer: text=",
        text,
        'socket=',
        !!socket,
        'companyId=',
        companyId
      );
      return;
    }

    const localId = uuidv4();
    const newMsg = {
      id: localId,
      company_id: companyId,
      content: text,
      timestamp: new Date().toISOString(),
      sender_name: myName,
      sender_role: 'company',
      _localId: localId,
    };

    console.log('[ChatWidget] 📤 Envoi message:', newMsg);
    setMessages((prev) => [...prev, newMsg]);
    localIdsRef.current.add(localId);
    setInput('');

    socket.emit('team_chat_message', newMsg, (response) => {
      console.log('[ChatWidget] 📥 Réponse du serveur:', response);
      if (response && response.error) {
        console.error('[ChatWidget] ❌ Erreur serveur:', response.error);
      }
    });
  };

  // Construction de la liste avec séparateurs de date
  const renderMessages = () => {
    let lastDate = null;
    return messages.map((msg) => {
      const msgDate = formatDate(msg.timestamp);
      const showDateSeparator = msgDate !== lastDate;
      lastDate = msgDate;
      // ✅ Normaliser: backend envoie "COMPANY"/"DRIVER"
      const isSent = (msg.sender_role || '').toString().toUpperCase() === 'COMPANY';

      return (
        <React.Fragment key={msg.id + (showDateSeparator ? '-sep' : '')}>
          {showDateSeparator && <div className="chat-date-separator">{msgDate}</div>}
          <div className={`chat-message ${isSent ? 'sent' : 'received'}`}>
            <div className="chat-author">
              {isSent ? myName : msg.sender_name}
              {msg.receiver_name && <span className="chat-receiver"> à {msg.receiver_name}</span>}
            </div>
            <div className="chat-content">{msg.content}</div>
            <div className="chat-time">{new Date(msg.timestamp).toLocaleTimeString()}</div>
          </div>
        </React.Fragment>
      );
    });
  };

  return (
    <>
      <button
        className="chat-widget-button"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? 'Fermer le chat' : 'Ouvrir le chat'}
      >
        {open ? <FiX size={24} /> : <FiMessageSquare size={24} />}
      </button>

      {open && (
        <div className="chat-widget-popup">
          <div className="chat-messages" ref={listRef}>
            {renderMessages()}
          </div>
          <div className="chat-input">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Écrire un message..."
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              aria-label="Champ de saisie du message"
            />
            <button onClick={sendMessage} disabled={!input.trim()}>
              Envoyer
            </button>
          </div>
        </div>
      )}
    </>
  );
}
