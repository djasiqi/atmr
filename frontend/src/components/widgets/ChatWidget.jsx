// src/components/widgets/ChatWidget.jsx
// ✅ Version alignée avec le chat mobile (WhatsApp-like)

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { FiMessageSquare, FiX, FiPaperclip, FiSend, FiImage, FiFileText } from 'react-icons/fi';
import useCompanySocket from '../../hooks/useCompanySocket';
import apiClient from '../../utils/apiClient';
import { v4 as uuidv4 } from 'uuid';
import { jwtDecode } from 'jwt-decode';
import './ChatWidget.css';

export default function ChatWidget({ companyId }) {
  const socket = useCompanySocket();
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [hasMore, setHasMore] = useState(true);
  const [oldestTs, setOldestTs] = useState(null);
  const [isTeamTyping, setIsTeamTyping] = useState(false);
  const [showAttachmentMenu, setShowAttachmentMenu] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [pdfPreview, setPdfPreview] = useState(null);
  const listRef = useRef();
  const localIdsRef = useRef(new Set());
  const typingTimeoutRef = useRef(null);
  const isAtBottomRef = useRef(true);
  const hasDoneInitialScrollRef = useRef(false);

  // Récupérer l'ID de l'utilisateur actuel depuis le token
  const currentUserId = useMemo(() => {
    try {
      const token = localStorage.getItem('authToken');
      if (!token) return null;
      const decoded = jwtDecode(token);
      // Le backend envoie user_id dans le token, ou on peut utiliser sub/public_id
      return decoded.user_id || decoded.sub || null;
    } catch {
      return null;
    }
  }, []);

  const [myName] = useState(() => {
    try {
      const user = JSON.parse(localStorage.getItem('user') || '{}');
      return user.username || 'Moi';
    } catch {
      return 'Moi';
    }
  });

  // Helper pour formater la date au format "le DD.MM.YYYY"
  const formatDate = (iso) => {
    const d = new Date(iso);
    const day = String(d.getDate()).padStart(2, '0');
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const year = d.getFullYear();
    return `le ${day}.${month}.${year}`;
  };

  // Helper pour obtenir les initiales d'un nom
  const getInitials = (name) => {
    if (!name) return '?';
    const parts = name.trim().split(/\s+/);
    if (parts.length >= 2) {
      return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
  };

  // Helper pour obtenir une couleur basée sur le nom
  const getColorForName = (name) => {
    if (!name) return '#0A7F59';
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
      hash = name.charCodeAt(i) + ((hash << 5) - hash);
    }
    const colors = [
      '#0A7F59', '#5F7369', '#8B4513', '#4169E1', '#DC143C',
      '#FF6347', '#32CD32', '#FFD700', '#9370DB', '#20B2AA'
    ];
    return colors[Math.abs(hash) % colors.length];
  };

  // Helper pour formater la taille des fichiers
  const formatSize = (bytes) => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  // Scroll vers le bas
  const scrollToBottom = useCallback((smooth = true) => {
    if (listRef.current) {
      listRef.current.scrollTo({
        top: listRef.current.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto',
      });
      isAtBottomRef.current = true;
    }
  }, []);

  // 1️⃣ Chargement initial (20 derniers messages)
  useEffect(() => {
    if (!open || !companyId) return;
    hasDoneInitialScrollRef.current = false;
    apiClient
      .get(`/messages/${companyId}`, { params: { limit: 20 } })
      .then((res) => {
        const raw = Array.isArray(res.data.messages) ? res.data.messages : res.data;
        const formatted = raw.map((m) => ({
          id: m.id,
          sender_id: m.sender_id,
          company_id: companyId,
          content: m.content,
          timestamp: m.timestamp,
          sender_name: m.sender_name || m.sender,
          sender_role: m.sender_role,
          image_url: m.image_url || m.image,
          pdf_url: m.pdf_url || m.pdf,
          pdf_filename: m.pdf_filename,
          pdf_size: m.pdf_size,
          _localId: null,
        }));
        // Trier par timestamp (plus ancien en premier)
        const sorted = formatted.sort((a, b) => {
          const timeA = new Date(a.timestamp || 0).getTime();
          const timeB = new Date(b.timestamp || 0).getTime();
          return timeA - timeB;
        });
        setMessages(sorted);
        if (sorted.length < 20) setHasMore(false);
        if (sorted.length > 0) setOldestTs(sorted[0].timestamp);
        // Scroll vers le bas après le chargement
        setTimeout(() => {
          scrollToBottom(false);
          hasDoneInitialScrollRef.current = true;
        }, 100);
      })
      .catch((err) => console.error('❌ Erreur loading messages:', err));
  }, [open, companyId, scrollToBottom]);

  // 2️⃣ Scroll automatique vers le bas pour nouveaux messages si on est déjà en bas
  useEffect(() => {
    if (messages.length && listRef.current && hasDoneInitialScrollRef.current && isAtBottomRef.current) {
      scrollToBottom(true);
    }
  }, [messages.length, scrollToBottom]);

  // 3️⃣ Infinite scroll ascendant pour loader les anciens
  useEffect(() => {
    const el = listRef.current;
    if (!el || !hasMore) return;

    const onScroll = () => {
      // Vérifier si on est en bas
      const isBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      isAtBottomRef.current = isBottom;

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
              sender_id: m.sender_id,
              company_id: companyId,
              content: m.content,
              timestamp: m.timestamp,
              sender_name: m.sender_name || m.sender,
              sender_role: m.sender_role,
              image_url: m.image_url || m.image,
              pdf_url: m.pdf_url || m.pdf,
              pdf_filename: m.pdf_filename,
              pdf_size: m.pdf_size,
              _localId: null,
            }));
            // Trier et dédupliquer
            setMessages((prev) => {
              const combined = [...formatted, ...prev];
              const unique = combined.filter((msg, idx, arr) => 
                arr.findIndex(m => m.id === msg.id) === idx
              );
              return unique.sort((a, b) => {
                const timeA = new Date(a.timestamp || 0).getTime();
                const timeB = new Date(b.timestamp || 0).getTime();
                return timeA - timeB;
              });
            });
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
    if (!socket || !companyId) return;

    const handleMessage = (msg) => {
      // Vérifier si c'est un message local (optimistic update)
      if (msg._localId && localIdsRef.current.has(msg._localId)) {
        localIdsRef.current.delete(msg._localId);
        // Remplacer le message local par le message du serveur
        setMessages((prev) => {
          const filtered = prev.filter((m) => m._localId !== msg._localId);
          const updated = [...filtered, msg];
          return updated.sort((a, b) => {
            const timeA = new Date(a.timestamp || 0).getTime();
            const timeB = new Date(b.timestamp || 0).getTime();
            return timeA - timeB;
          });
        });
        return;
      }
      // Vérifier si le message existe déjà (éviter les doublons)
      setMessages((prev) => {
        const exists = prev.some(
          (m) =>
            (m.id && msg.id && m.id === msg.id) ||
            (m._localId && msg._localId && m._localId === msg._localId)
        );
        if (exists) return prev;
        const updated = [...prev, msg];
        return updated.sort((a, b) => {
          const timeA = new Date(a.timestamp || 0).getTime();
          const timeB = new Date(b.timestamp || 0).getTime();
          return timeA - timeB;
        });
      });
    };

    const handleTypingStart = () => setIsTeamTyping(true);
    const handleTypingStop = () => setIsTeamTyping(false);

    socket.on('team_chat_message', handleMessage);
    socket.on('typing_start', handleTypingStart);
    socket.on('typing_stop', handleTypingStop);
    socket.on('error', (error) => console.error('❌ Erreur WebSocket:', error));

    return () => {
      socket.off('team_chat_message', handleMessage);
      socket.off('typing_start', handleTypingStart);
      socket.off('typing_stop', handleTypingStop);
      socket.off('error');
    };
  }, [socket, companyId]);

  // 5️⃣ Envoi d'un message (optimistic update)
  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text || !socket || !companyId) return;

    const localId = uuidv4();
    const newMsg = {
      id: localId,
      sender_id: currentUserId,
      company_id: companyId,
      content: text,
      timestamp: new Date().toISOString(),
      sender_name: myName,
      sender_role: 'COMPANY',
      _localId: localId,
    };

    setMessages((prev) => {
      const updated = [...prev, newMsg];
      return updated.sort((a, b) => {
        const timeA = new Date(a.timestamp || 0).getTime();
        const timeB = new Date(b.timestamp || 0).getTime();
        return timeA - timeB;
      });
    });
    localIdsRef.current.add(localId);
    setInput('');
    scrollToBottom(true);

    // Arrêter le typing indicator
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    socket.emit('typing_stop');

    socket.emit('team_chat_message', { content: text, receiver_id: null }, (response) => {
      if (response && response.error) {
        console.error('❌ Erreur serveur:', response.error);
      }
    });
  }, [input, socket, companyId, currentUserId, myName, scrollToBottom]);

  // Typing indicator
  const handleTyping = useCallback((text) => {
    setInput(text);
    if (!socket) return;
    socket.emit('typing_start');
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    typingTimeoutRef.current = setTimeout(() => {
      socket.emit('typing_stop');
    }, 900);
  }, [socket]);

  // Envoi d'image
  const handleSendImage = useCallback(async (file) => {
    if (!socket || !companyId) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const uploadRes = await apiClient.post('/messages/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const { url } = uploadRes.data;
      socket.emit('team_chat_message', {
        content: '',
        image_url: url,
        receiver_id: null,
      });
    } catch (error) {
      console.error('❌ Erreur upload image:', error);
    }
  }, [socket, companyId]);

  // Envoi de PDF
  const handleSendPdf = useCallback(async (file) => {
    if (!socket || !companyId) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const uploadRes = await apiClient.post('/messages/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const { url, filename, size_bytes } = uploadRes.data;
      socket.emit('team_chat_message', {
        content: '',
        pdf_url: url,
        pdf_filename: filename,
        pdf_size: size_bytes,
        receiver_id: null,
      });
    } catch (error) {
      console.error('❌ Erreur upload PDF:', error);
    }
  }, [socket, companyId]);

  // Construction de la liste avec séparateurs de date
  const listItemsWithDates = useMemo(() => {
    const items = [];
    let lastDate = null;
    messages.forEach((msg) => {
      const msgDate = formatDate(msg.timestamp);
      if (msgDate !== lastDate) {
        items.push({ type: 'dateSeparator', date: msgDate });
        lastDate = msgDate;
      }
      items.push({ type: 'message', message: msg });
    });
    return items;
  }, [messages]);

  // Déterminer si un message est envoyé par l'utilisateur actuel
  const isOwnMessage = useCallback((msg) => {
    if (!currentUserId || !msg.sender_id) return false;
    return Number(msg.sender_id) === Number(currentUserId);
  }, [currentUserId]);

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
            {listItemsWithDates.length === 0 ? (
              <div className="chat-empty">
                <FiMessageSquare size={48} />
                <p>Aucun message pour le moment.</p>
                <p>Commencez la conversation avec votre équipe !</p>
              </div>
            ) : (
              listItemsWithDates.map((item, idx) => {
                if (item.type === 'dateSeparator') {
                  return (
                    <div key={`date-${item.date}-${idx}`} className="chat-date-separator">
                      <span className="chat-date-line"></span>
                      <span className="chat-date-text">{item.date}</span>
                      <span className="chat-date-line"></span>
                    </div>
                  );
                }
                const msg = item.message;
                const isOwn = isOwnMessage(msg);
                return (
                  <div
                    key={msg.id || msg._localId || `msg-${idx}`}
                    className={`chat-message-wrapper ${isOwn ? 'sent' : 'received'}`}
                  >
                    {!isOwn && (
                      <div
                        className="chat-avatar"
                        style={{ backgroundColor: getColorForName(msg.sender_name) }}
                      >
                        {msg.sender_name ? getInitials(msg.sender_name) : '?'}
                      </div>
                    )}
                    <div className={`chat-message ${isOwn ? 'sent' : 'received'}`}>
                      {msg.image_url && (
                        <div
                          className="chat-image"
                          onClick={() => setImagePreview(msg.image_url)}
                        >
                          <img src={msg.image_url} alt="Image" />
                        </div>
                      )}
                      {msg.pdf_url && (
                        <div
                          className="chat-pdf"
                          onClick={() => setPdfPreview(msg.pdf_url)}
                        >
                          <FiFileText size={28} />
                          <div>
                            <div className="chat-pdf-name">{msg.pdf_filename || 'Document PDF'}</div>
                            {msg.pdf_size && (
                              <div className="chat-pdf-size">{formatSize(msg.pdf_size)}</div>
                            )}
                          </div>
                        </div>
                      )}
                      {msg.content && <div className="chat-content">{msg.content}</div>}
                      <div className={`chat-time ${isOwn ? 'sent' : 'received'}`}>
                        {new Date(msg.timestamp).toLocaleTimeString('fr-FR', {
                          hour: '2-digit',
                          minute: '2-digit',
                          hour12: false,
                        })}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
            {isTeamTyping && (
              <div className="chat-typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            )}
          </div>
          <div className="chat-input">
            {showAttachmentMenu && (
              <div className="chat-attachment-menu">
                <button onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.accept = 'image/*';
                  input.onchange = (e) => {
                    const file = e.target.files[0];
                    if (file) handleSendImage(file);
                    setShowAttachmentMenu(false);
                  };
                  input.click();
                }}>
                  <FiImage size={20} />
                  <span>Photo</span>
                </button>
                <button onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.accept = 'application/pdf';
                  input.onchange = (e) => {
                    const file = e.target.files[0];
                    if (file) handleSendPdf(file);
                    setShowAttachmentMenu(false);
                  };
                  input.click();
                }}>
                  <FiFileText size={20} />
                  <span>PDF</span>
                </button>
              </div>
            )}
            <button
              className="chat-attach-button"
              onClick={() => setShowAttachmentMenu(!showAttachmentMenu)}
            >
              <FiPaperclip size={20} />
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => handleTyping(e.target.value)}
              placeholder="Écrire un message..."
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              aria-label="Champ de saisie du message"
            />
            <button
              className="chat-send-button"
              onClick={sendMessage}
              disabled={!input.trim()}
            >
              <FiSend size={18} />
            </button>
          </div>
        </div>
      )}

      {/* Modals pour preview */}
      {imagePreview && (
        <div className="chat-modal-overlay" onClick={() => setImagePreview(null)}>
          <div className="chat-modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="chat-modal-close" onClick={() => setImagePreview(null)}>
              <FiX size={24} />
            </button>
            <img src={imagePreview} alt="Preview" />
          </div>
        </div>
      )}

      {pdfPreview && (
        <div className="chat-modal-overlay" onClick={() => setPdfPreview(null)}>
          <div className="chat-modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="chat-modal-close" onClick={() => setPdfPreview(null)}>
              <FiX size={24} />
            </button>
            <iframe src={pdfPreview} style={{ width: '100%', height: '80vh', border: 'none' }} />
          </div>
        </div>
      )}
    </>
  );
}
