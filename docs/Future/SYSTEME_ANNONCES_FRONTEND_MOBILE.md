# 📱 Système d'annonces - Frontend & Mobile (Suite)

**Document complémentaire à**: `SYSTEME_ANNONCES_IMPLEMENTATION.md`  
**Date**: 2026-01-13  
**Version**: 1.0

---

## 📋 Table des matières

1. [Phase 4: Frontend Entreprise Web](#phase-4-frontend-entreprise-web)
2. [Phase 5: Frontend Entreprise Mobile](#phase-5-frontend-entreprise-mobile)
3. [Phase 6: Frontend Chauffeur Mobile](#phase-6-frontend-chauffeur-mobile)
4. [Phase 7: Socket.IO temps réel](#phase-7-socketio-temps-réel)
5. [Phase 8: Push Notifications](#phase-8-push-notifications)
6. [Phase 9: Tests](#phase-9-tests)
7. [Phase 10: Déploiement](#phase-10-déploiement)

---

## 🌐 Phase 4: Frontend Entreprise Web

### Étape 4.1: Créer le composant AnnouncementBanner

**Fichier**: `frontend/src/components/announcements/AnnouncementBanner.jsx`

```jsx
// frontend/src/components/announcements/AnnouncementBanner.jsx
import React, { useState, useEffect } from "react";
import { FaTimes, FaExternalLinkAlt } from "react-icons/fa";
import axios from "axios";
import { logger } from "../../utils/logger";
import styles from "./AnnouncementBanner.module.css";

const ANNOUNCEMENT_ICONS = {
  info: "📘",
  warning: "⚠️",
  maintenance: "🔧",
  update: "🚀",
  emergency: "🚨",
};

const AnnouncementBanner = () => {
  const [announcements, setAnnouncements] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnnouncements();
  }, []);

  const loadAnnouncements = async () => {
    try {
      const response = await axios.get("/api/v1/announcements");
      setAnnouncements(response.data.announcements || []);
      logger.debug(
        "[AnnouncementBanner] Loaded announcements:",
        response.data.announcements
      );
    } catch (error) {
      logger.error("[AnnouncementBanner] Error loading announcements:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDismiss = async (announcementId) => {
    try {
      await axios.post(`/api/v1/announcements/${announcementId}/dismiss`);

      // Retirer l'annonce de l'affichage
      setAnnouncements((prev) =>
        prev.filter((ann) => ann.id !== announcementId)
      );

      logger.info(
        "[AnnouncementBanner] Announcement dismissed:",
        announcementId
      );
    } catch (error) {
      logger.error(
        "[AnnouncementBanner] Error dismissing announcement:",
        error
      );
    }
  };

  if (loading) {
    return null;
  }

  if (announcements.length === 0) {
    return null;
  }

  return (
    <div className={styles.container}>
      {announcements.map((announcement) => (
        <div
          key={announcement.id}
          className={styles.banner}
          data-type={announcement.type}
          data-priority={announcement.priority}
        >
          <div className={styles.content}>
            <div className={styles.icon}>
              {ANNOUNCEMENT_ICONS[announcement.type] || "📢"}
            </div>

            <div className={styles.text}>
              <div className={styles.title}>{announcement.title}</div>
              <div className={styles.message}>{announcement.message}</div>
            </div>

            {announcement.action_button_text &&
              announcement.action_button_url && (
                <a
                  href={announcement.action_button_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={styles.actionButton}
                >
                  {announcement.action_button_text}
                  <FaExternalLinkAlt size={12} />
                </a>
              )}
          </div>

          <button
            className={styles.closeButton}
            onClick={() => handleDismiss(announcement.id)}
            aria-label="Fermer l'annonce"
          >
            <FaTimes />
          </button>
        </div>
      ))}
    </div>
  );
};

export default AnnouncementBanner;
```

**Fichier**: `frontend/src/components/announcements/AnnouncementBanner.module.css`

```css
/* frontend/src/components/announcements/AnnouncementBanner.module.css */

.container {
  position: fixed;
  top: 80px; /* Sous le header */
  left: 0;
  right: 0;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 0 16px;
  pointer-events: none;
}

.banner {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 16px 20px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  animation: slideDown 0.3s ease-out;
  pointer-events: auto;
  backdrop-filter: blur(10px);
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

@keyframes slideDown {
  from {
    transform: translateY(-20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

/* Styles par type */
.banner[data-type="info"] {
  background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
  border-left: 4px solid #2196f3;
}

.banner[data-type="warning"] {
  background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
  border-left: 4px solid #ff9800;
}

.banner[data-type="maintenance"] {
  background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
  border-left: 4px solid #9c27b0;
}

.banner[data-type="update"] {
  background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
  border-left: 4px solid #4caf50;
}

.banner[data-type="emergency"] {
  background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
  border-left: 4px solid #f44336;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%,
  100% {
    box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3);
  }
  50% {
    box-shadow: 0 4px 20px rgba(244, 67, 54, 0.6);
  }
}

/* Styles par priorité (surcharge) */
.banner[data-priority="critical"] {
  font-weight: 600;
  border-left-width: 6px;
}

.content {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;
}

.icon {
  font-size: 32px;
  flex-shrink: 0;
}

.text {
  flex: 1;
}

.title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 4px;
  color: #1a1a1a;
}

.message {
  font-size: 14px;
  line-height: 1.5;
  color: #4a4a4a;
}

.actionButton {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: white;
  border: 2px solid currentColor;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  color: inherit;
  transition: all 0.2s;
  flex-shrink: 0;
}

.actionButton:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.closeButton {
  padding: 8px;
  background: rgba(0, 0, 0, 0.1);
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
  flex-shrink: 0;
  color: #4a4a4a;
}

.closeButton:hover {
  background: rgba(0, 0, 0, 0.2);
}

/* Responsive */
@media (max-width: 768px) {
  .container {
    top: 60px;
    padding: 0 8px;
  }

  .banner {
    flex-direction: column;
    gap: 12px;
    padding: 12px 16px;
  }

  .content {
    flex-direction: column;
    align-items: flex-start;
  }

  .icon {
    font-size: 24px;
  }

  .title {
    font-size: 15px;
  }

  .message {
    font-size: 13px;
  }

  .actionButton {
    width: 100%;
    justify-content: center;
  }

  .closeButton {
    position: absolute;
    top: 12px;
    right: 12px;
  }
}
```

### Étape 4.2: Ajouter le hook useAnnouncements

**Fichier**: `frontend/src/hooks/useAnnouncements.js`

```javascript
// frontend/src/hooks/useAnnouncements.js
import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { getCompanySocket } from "../services/companySocket";
import { logger } from "../utils/logger";

/**
 * Hook pour gérer les annonces système
 * - Récupère les annonces depuis l'API
 * - Écoute les nouvelles annonces via Socket.IO
 * - Gère la fermeture des annonces
 */
export const useAnnouncements = () => {
  const [announcements, setAnnouncements] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Récupérer les annonces depuis l'API
  const fetchAnnouncements = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get("/api/v1/announcements");
      setAnnouncements(response.data.announcements || []);

      logger.debug(
        "[useAnnouncements] Loaded announcements:",
        response.data.announcements
      );
    } catch (err) {
      logger.error("[useAnnouncements] Error fetching announcements:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fermer une annonce
  const dismissAnnouncement = useCallback(async (announcementId) => {
    try {
      await axios.post(`/api/v1/announcements/${announcementId}/dismiss`);

      // Retirer l'annonce localement
      setAnnouncements((prev) =>
        prev.filter((ann) => ann.id !== announcementId)
      );

      logger.info("[useAnnouncements] Announcement dismissed:", announcementId);
      return true;
    } catch (err) {
      logger.error("[useAnnouncements] Error dismissing announcement:", err);
      return false;
    }
  }, []);

  // Écouter les nouvelles annonces via Socket.IO
  useEffect(() => {
    fetchAnnouncements();

    const socket = getCompanySocket();
    if (!socket) {
      logger.warn("[useAnnouncements] Socket not available");
      return;
    }

    // Handler pour les nouvelles annonces
    const handleNewAnnouncement = (data) => {
      logger.info("[useAnnouncements] New announcement received:", data);

      // Ajouter l'annonce si elle n'existe pas déjà
      setAnnouncements((prev) => {
        const exists = prev.some((ann) => ann.id === data.id);
        if (exists) {
          return prev;
        }
        return [data, ...prev];
      });

      // Afficher une notification toast (optionnel)
      if (data.priority === "critical" || data.type === "emergency") {
        // Notification sonore ou visuelle pour les annonces critiques
        if ("Notification" in window && Notification.permission === "granted") {
          new Notification(data.title, {
            body: data.message,
            icon: "/favicon.ico",
            tag: `announcement-${data.id}`,
          });
        }
      }
    };

    // Handler pour les annonces supprimées
    const handleAnnouncementRemoved = (data) => {
      logger.info("[useAnnouncements] Announcement removed:", data);
      setAnnouncements((prev) =>
        prev.filter((ann) => ann.id !== data.announcement_id)
      );
    };

    // Écouter les événements
    socket.on("system_announcement", handleNewAnnouncement);
    socket.on("announcement_removed", handleAnnouncementRemoved);

    // Cleanup
    return () => {
      socket.off("system_announcement", handleNewAnnouncement);
      socket.off("announcement_removed", handleAnnouncementRemoved);
    };
  }, [fetchAnnouncements]);

  return {
    announcements,
    loading,
    error,
    dismissAnnouncement,
    refetch: fetchAnnouncements,
  };
};
```

### Étape 4.3: Intégrer dans le Dashboard Entreprise

**Fichier**: `frontend/src/pages/company/Dashboard/CompanyDashboard.jsx`

```jsx
// Ajouter l'import
import AnnouncementBanner from "../../../components/announcements/AnnouncementBanner";

// Dans le composant, ajouter avant le contenu principal:
const CompanyDashboard = () => {
  // ... autres hooks ...

  return (
    <div className={styles.dashboardContainer}>
      <HeaderDashboard />
      <CompanySidebar />

      {/* Annonces système */}
      <AnnouncementBanner />

      {/* Contenu principal */}
      <main className={styles.mainContent}>
        {/* ... reste du contenu ... */}
      </main>
    </div>
  );
};
```

### Étape 4.4: Demander la permission pour les notifications

**Fichier**: `frontend/src/utils/notifications.js`

```javascript
// frontend/src/utils/notifications.js
import { logger } from "./logger";

/**
 * Demande la permission pour les notifications du navigateur
 */
export const requestNotificationPermission = async () => {
  if (!("Notification" in window)) {
    logger.warn("[Notifications] Notifications not supported in this browser");
    return false;
  }

  if (Notification.permission === "granted") {
    logger.debug("[Notifications] Permission already granted");
    return true;
  }

  if (Notification.permission === "denied") {
    logger.warn("[Notifications] Permission denied");
    return false;
  }

  try {
    const permission = await Notification.requestPermission();
    logger.info("[Notifications] Permission:", permission);
    return permission === "granted";
  } catch (error) {
    logger.error("[Notifications] Error requesting permission:", error);
    return false;
  }
};

/**
 * Affiche une notification du navigateur
 */
export const showNotification = (title, options = {}) => {
  if (Notification.permission !== "granted") {
    logger.warn("[Notifications] Permission not granted");
    return;
  }

  try {
    const notification = new Notification(title, {
      icon: "/favicon.ico",
      badge: "/favicon.ico",
      ...options,
    });

    notification.onclick = () => {
      window.focus();
      notification.close();
      if (options.onClick) {
        options.onClick();
      }
    };

    return notification;
  } catch (error) {
    logger.error("[Notifications] Error showing notification:", error);
  }
};
```

**Appeler au démarrage de l'app**:

```jsx
// frontend/src/App.js
import { requestNotificationPermission } from "./utils/notifications";

// Dans useEffect au montage:
useEffect(() => {
  // Demander la permission après quelques secondes (UX)
  const timer = setTimeout(() => {
    requestNotificationPermission();
  }, 3000);

  return () => clearTimeout(timer);
}, []);
```

---

## 📱 Phase 5: Frontend Entreprise Mobile

### Étape 5.1: Créer le composant AnnouncementBanner (Mobile)

**Fichier**: `mobile/operations-app/components/announcements/AnnouncementBanner.tsx`

```typescript
// mobile/operations-app/components/announcements/AnnouncementBanner.tsx
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Linking,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import axios from "axios";
import { palette } from "../../theme/palette";
import { logger } from "../../utils/logger";
import { getStandardAPIURL } from "../../config/api";

const ANNOUNCEMENT_ICONS = {
  info: "information-circle",
  warning: "warning",
  maintenance: "construct",
  update: "rocket",
  emergency: "alert-circle",
} as const;

const ANNOUNCEMENT_COLORS = {
  info: { bg: "#E3F2FD", border: "#2196F3", text: "#1565C0" },
  warning: { bg: "#FFF3E0", border: "#FF9800", text: "#E65100" },
  maintenance: { bg: "#F3E5F5", border: "#9C27B0", text: "#6A1B9A" },
  update: { bg: "#E8F5E9", border: "#4CAF50", text: "#2E7D32" },
  emergency: { bg: "#FFEBEE", border: "#F44336", text: "#C62828" },
};

interface Announcement {
  id: number;
  title: string;
  message: string;
  type: "info" | "warning" | "maintenance" | "update" | "emergency";
  priority: "low" | "normal" | "high" | "critical";
  action_button_text?: string;
  action_button_url?: string;
}

export const AnnouncementBanner: React.FC = () => {
  const [announcements, setAnnouncements] = useState<Announcement[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnnouncements();
  }, []);

  const loadAnnouncements = async () => {
    try {
      const apiURL = getStandardAPIURL();
      const response = await axios.get(`${apiURL}/announcements`);

      setAnnouncements(response.data.announcements || []);
      logger.debug(
        "[AnnouncementBanner] Loaded announcements:",
        response.data.announcements
      );
    } catch (error) {
      logger.error("[AnnouncementBanner] Error loading announcements:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDismiss = async (announcementId: number) => {
    try {
      const apiURL = getStandardAPIURL();
      await axios.post(`${apiURL}/announcements/${announcementId}/dismiss`);

      setAnnouncements((prev) =>
        prev.filter((ann) => ann.id !== announcementId)
      );
      logger.info(
        "[AnnouncementBanner] Announcement dismissed:",
        announcementId
      );
    } catch (error) {
      logger.error(
        "[AnnouncementBanner] Error dismissing announcement:",
        error
      );
    }
  };

  const handleActionPress = async (url: string) => {
    try {
      const supported = await Linking.canOpenURL(url);
      if (supported) {
        await Linking.openURL(url);
      }
    } catch (error) {
      logger.error("[AnnouncementBanner] Error opening URL:", error);
    }
  };

  if (loading || announcements.length === 0) {
    return null;
  }

  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={{
        paddingHorizontal: 16,
        paddingVertical: 8,
        gap: 12,
      }}
    >
      {announcements.map((announcement) => {
        const colors = ANNOUNCEMENT_COLORS[announcement.type];
        const iconName = ANNOUNCEMENT_ICONS[announcement.type];

        return (
          <View
            key={announcement.id}
            style={{
              backgroundColor: colors.bg,
              borderLeftWidth: 4,
              borderLeftColor: colors.border,
              borderRadius: 12,
              padding: 16,
              width: 320,
              maxWidth: "90%",
            }}
          >
            <View
              style={{
                flexDirection: "row",
                justifyContent: "space-between",
                alignItems: "flex-start",
                marginBottom: 8,
              }}
            >
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 8,
                  flex: 1,
                }}
              >
                <Ionicons name={iconName} size={24} color={colors.text} />
                <Text
                  style={{
                    fontSize: 16,
                    fontWeight: "600",
                    color: colors.text,
                    flex: 1,
                  }}
                >
                  {announcement.title}
                </Text>
              </View>

              <TouchableOpacity
                onPress={() => handleDismiss(announcement.id)}
                style={{ padding: 4 }}
              >
                <Ionicons name="close" size={20} color={colors.text} />
              </TouchableOpacity>
            </View>

            <Text
              style={{
                fontSize: 14,
                lineHeight: 20,
                color: colors.text,
                marginBottom: 12,
              }}
            >
              {announcement.message}
            </Text>

            {announcement.action_button_text &&
              announcement.action_button_url && (
                <TouchableOpacity
                  onPress={() =>
                    handleActionPress(announcement.action_button_url!)
                  }
                  style={{
                    backgroundColor: "white",
                    borderWidth: 2,
                    borderColor: colors.border,
                    borderRadius: 8,
                    paddingVertical: 8,
                    paddingHorizontal: 12,
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 6,
                  }}
                >
                  <Text
                    style={{
                      fontSize: 14,
                      fontWeight: "500",
                      color: colors.text,
                    }}
                  >
                    {announcement.action_button_text}
                  </Text>
                  <Ionicons name="open-outline" size={14} color={colors.text} />
                </TouchableOpacity>
              )}
          </View>
        );
      })}
    </ScrollView>
  );
};
```

### Étape 5.2: Créer le hook useAnnouncements (Mobile)

**Fichier**: `mobile/operations-app/hooks/useAnnouncements.ts`

```typescript
// mobile/operations-app/hooks/useAnnouncements.ts
import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { getStandardAPIURL } from "../config/api";
import { logger } from "../utils/logger";
import { useEnterpriseSocket } from "./useEnterpriseSocket";

interface Announcement {
  id: number;
  title: string;
  message: string;
  type: "info" | "warning" | "maintenance" | "update" | "emergency";
  priority: "low" | "normal" | "high" | "critical";
  action_button_text?: string;
  action_button_url?: string;
  start_date: string;
  end_date?: string;
}

export const useAnnouncements = () => {
  const [announcements, setAnnouncements] = useState<Announcement[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const socket = useEnterpriseSocket();

  // Récupérer les annonces depuis l'API
  const fetchAnnouncements = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const apiURL = getStandardAPIURL();
      const response = await axios.get<{ announcements: Announcement[] }>(
        `${apiURL}/announcements`
      );

      setAnnouncements(response.data.announcements || []);
      logger.debug(
        "[useAnnouncements] Loaded announcements:",
        response.data.announcements
      );
    } catch (err) {
      logger.error("[useAnnouncements] Error fetching announcements:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  // Fermer une annonce
  const dismissAnnouncement = useCallback(async (announcementId: number) => {
    try {
      const apiURL = getStandardAPIURL();
      await axios.post(`${apiURL}/announcements/${announcementId}/dismiss`);

      setAnnouncements((prev) =>
        prev.filter((ann) => ann.id !== announcementId)
      );
      logger.info("[useAnnouncements] Announcement dismissed:", announcementId);
      return true;
    } catch (err) {
      logger.error("[useAnnouncements] Error dismissing announcement:", err);
      return false;
    }
  }, []);

  // Écouter les nouvelles annonces via Socket.IO
  useEffect(() => {
    fetchAnnouncements();

    if (!socket) {
      logger.warn("[useAnnouncements] Socket not available");
      return;
    }

    // Handler pour les nouvelles annonces
    const handleNewAnnouncement = (data: Announcement) => {
      logger.info("[useAnnouncements] New announcement received:", data);

      setAnnouncements((prev) => {
        const exists = prev.some((ann) => ann.id === data.id);
        if (exists) return prev;
        return [data, ...prev];
      });
    };

    // Handler pour les annonces supprimées
    const handleAnnouncementRemoved = (data: { announcement_id: number }) => {
      logger.info("[useAnnouncements] Announcement removed:", data);
      setAnnouncements((prev) =>
        prev.filter((ann) => ann.id !== data.announcement_id)
      );
    };

    socket.on("system_announcement", handleNewAnnouncement);
    socket.on("announcement_removed", handleAnnouncementRemoved);

    return () => {
      socket.off("system_announcement", handleNewAnnouncement);
      socket.off("announcement_removed", handleAnnouncementRemoved);
    };
  }, [socket, fetchAnnouncements]);

  return {
    announcements,
    loading,
    error,
    dismissAnnouncement,
    refetch: fetchAnnouncements,
  };
};
```

### Étape 5.3: Intégrer dans le Dashboard Mobile

**Fichier**: `mobile/operations-app/app/(enterprise)/index.tsx`

```typescript
// Ajouter l'import
import { AnnouncementBanner } from "../../components/announcements/AnnouncementBanner";

// Dans le composant:
export default function EnterpriseDashboard() {
  return (
    <View style={{ flex: 1 }}>
      {/* Header */}
      <Header />

      {/* Annonces système */}
      <AnnouncementBanner />

      {/* Contenu principal */}
      <ScrollView style={{ flex: 1 }}>
        {/* ... contenu du dashboard ... */}
      </ScrollView>
    </View>
  );
}
```

---

## 🚗 Phase 6: Frontend Chauffeur Mobile

### Étape 6.1: Adapter le composant pour chauffeurs

**Fichier**: `mobile/driver-app/components/announcements/AnnouncementBanner.tsx`

```typescript
// mobile/driver-app/components/announcements/AnnouncementBanner.tsx
// (Même code que pour entreprise mobile, adapter les imports selon votre structure)

import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Linking,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import axios from "axios";
import { logger } from "../../utils/logger";
import { API_URL } from "../../config/api";

// ... (même code que mobile/operations-app/components/announcements/AnnouncementBanner.tsx)
```

### Étape 6.2: Intégrer dans le Dashboard Chauffeur

```typescript
// mobile/driver-app/app/(driver)/index.tsx
import { AnnouncementBanner } from "../../components/announcements/AnnouncementBanner";

export default function DriverDashboard() {
  return (
    <View style={{ flex: 1 }}>
      <Header />
      <AnnouncementBanner />
      <ScrollView>{/* Contenu du dashboard chauffeur */}</ScrollView>
    </View>
  );
}
```

---

## 🔌 Phase 7: Socket.IO temps réel

### Étape 7.1: Configurer les rooms Socket.IO

**Fichier**: `backend/socket_handlers.py` (ou fichier existant gérant Socket.IO)

```python
# backend/socket_handlers.py
from flask_socketio import join_room, leave_room, emit
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from ext import socketio
from models import User, UserRole
import logging

logger = logging.getLogger(__name__)

@socketio.on("connect")
def handle_connect(auth):
    """Gestion de la connexion Socket.IO."""
    try:
        # Vérifier le token JWT
        verify_jwt_in_request()
        user_id = get_jwt_identity()

        user = User.query.get(user_id)
        if not user:
            logger.warning(f"❌ User {user_id} not found on socket connect")
            return False

        # Joindre les rooms appropriées selon le rôle
        if user.role == UserRole.COMPANY:
            join_room("companies")
            if user.company_id:
                join_room(f"company_{user.company_id}")
            logger.info(f"✅ Company user {user_id} joined socket rooms")

        elif user.role == UserRole.DRIVER:
            join_room("drivers")
            if user.company_id:
                join_room(f"company_{user.company_id}")
            logger.info(f"✅ Driver user {user_id} joined socket rooms")

        elif user.role == UserRole.ADMIN:
            join_room("admins")
            logger.info(f"✅ Admin user {user_id} joined socket rooms")

        emit("connected", {
            "status": "connected",
            "user_id": user_id,
            "role": user.role,
        })

        return True

    except Exception as e:
        logger.exception(f"❌ Socket connection error: {e}")
        return False

@socketio.on("disconnect")
def handle_disconnect():
    """Gestion de la déconnexion Socket.IO."""
    try:
        verify_jwt_in_request()
        user_id = get_jwt_identity()
        logger.info(f"👋 User {user_id} disconnected from socket")
    except Exception as e:
        logger.debug(f"Socket disconnect (no auth): {e}")
```

### Étape 7.2: Types TypeScript pour Socket.IO

**Fichier**: `frontend/src/types/socketEvents.ts`

```typescript
// Ajouter ces interfaces:

export interface SystemAnnouncementPayload {
  id: number;
  title: string;
  message: string;
  type: "info" | "warning" | "maintenance" | "update" | "emergency";
  priority: "low" | "normal" | "high" | "critical";
  start_date: string;
  end_date?: string;
  action_button_text?: string;
  action_button_url?: string;
  target_roles: string[];
  target_company_ids?: number[];
}

export interface AnnouncementRemovedPayload {
  announcement_id: number;
}

// Ajouter dans SocketEventMapServer:
export interface SocketEventMapServer {
  // ... autres événements ...
  system_announcement: SystemAnnouncementPayload;
  announcement_removed: AnnouncementRemovedPayload;
}
```

---

## 📲 Phase 8: Push Notifications

### Étape 8.1: Envoyer des push pour les annonces critiques

**Fichier**: `backend/services/announcements/push_service.py`

```python
# backend/services/announcements/push_service.py
"""Service pour envoyer des push notifications pour les annonces critiques."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from services.notifications.push import send_push_notification_to_user
from models import User, UserRole

if TYPE_CHECKING:
    from models import SystemAnnouncement

logger = logging.getLogger(__name__)


def send_announcement_push(announcement: SystemAnnouncement):
    """Envoie des push notifications pour une annonce critique.

    Envoie uniquement si:
    - Priorité = critical OU type = emergency
    - Utilisateurs ciblés par l'annonce

    Args:
        announcement: L'annonce à diffuser
    """
    # Ne push que pour les annonces critiques/urgentes
    if announcement.priority != "critical" and announcement.type != "emergency":
        logger.debug(
            f"[AnnouncementPush] Skipping push for announcement {announcement.id} "
            f"(priority={announcement.priority}, type={announcement.type})"
        )
        return

    try:
        # Récupérer les utilisateurs ciblés
        query = User.query.filter(User.is_active == True)

        # Filtrer par rôle
        if "all" not in announcement.target_roles:
            target_roles_enum = []
            for role_str in announcement.target_roles:
                if role_str.lower() == "company":
                    target_roles_enum.append(UserRole.COMPANY)
                elif role_str.lower() == "driver":
                    target_roles_enum.append(UserRole.DRIVER)

            if target_roles_enum:
                query = query.filter(User.role.in_(target_roles_enum))

        # Filtrer par entreprise si spécifié
        if announcement.target_company_ids:
            query = query.filter(User.company_id.in_(announcement.target_company_ids))

        users = query.all()

        logger.info(
            f"[AnnouncementPush] Sending push to {len(users)} users for announcement {announcement.id}"
        )

        # Envoyer une push à chaque utilisateur
        success_count = 0
        for user in users:
            try:
                result = send_push_notification_to_user(
                    user_id=user.id,
                    title=announcement.title,
                    body=announcement.message,
                    data={
                        "type": "announcement",
                        "announcement_id": str(announcement.id),
                        "announcement_type": announcement.type,
                        "priority": announcement.priority,
                    },
                )

                if result:
                    success_count += 1

            except Exception as e:
                logger.warning(
                    f"[AnnouncementPush] Failed to send push to user {user.id}: {e}"
                )
                continue

        logger.info(
            f"✅ [AnnouncementPush] Sent {success_count}/{len(users)} push notifications "
            f"for announcement {announcement.id}"
        )

    except Exception as e:
        logger.exception(
            f"❌ [AnnouncementPush] Error sending push for announcement {announcement.id}: {e}"
        )
```

### Étape 8.2: Intégrer dans la création d'annonce

**Fichier**: `backend/routes/announcements.py`

```python
# Ajouter l'import
from services.announcements.push_service import send_announcement_push

# Dans _broadcast_announcement(), ajouter:
def _broadcast_announcement(announcement: SystemAnnouncement):
    """Diffuse une annonce via Socket.IO et push notifications."""
    try:
        # ... (code Socket.IO existant) ...

        # Envoyer des push notifications pour les annonces critiques
        if announcement.priority == "critical" or announcement.type == "emergency":
            try:
                send_announcement_push(announcement)
            except Exception as e:
                logger.exception(f"❌ Erreur push announcement {announcement.id}: {e}")

    except Exception as e:
        logger.exception(f"❌ Erreur broadcast announcement {announcement.id}")
```

### Étape 8.3: Gérer les notifications sur mobile

**Fichier**: `mobile/operations-app/services/notification.ts`

```typescript
// Ajouter un handler spécifique pour les annonces
export const handleAnnouncementNotification = (
  notification: Notifications.Notification
) => {
  const data = notification.request.content.data;

  if (data.type === "announcement") {
    // Rediriger vers le dashboard ou afficher une modal
    logger.info("[Notifications] Announcement notification received:", data);

    // Exemple: Stocker l'ID pour l'afficher au démarrage
    AsyncStorage.setItem("pending_announcement_id", data.announcement_id);
  }
};
```

---

## ✅ Phase 9: Tests

### Étape 9.1: Tests unitaires Backend

**Fichier**: `backend/tests/test_announcements.py`

```python
# backend/tests/test_announcements.py
"""Tests pour les annonces système."""

import pytest
from datetime import UTC, datetime, timedelta
from models import SystemAnnouncement, User, UserRole


def test_create_announcement(client, admin_token):
    """Test de création d'une annonce."""
    response = client.post(
        "/api/v1/announcements",
        json={
            "title": "Test Announcement",
            "message": "This is a test",
            "type": "info",
            "priority": "normal",
            "start_date": datetime.now(UTC).isoformat(),
            "target_roles": ["all"],
            "is_published": False,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 201
    data = response.json
    assert data["title"] == "Test Announcement"
    assert data["is_published"] is False


def test_get_announcements_as_company(client, company_token):
    """Test de récupération des annonces pour une entreprise."""
    response = client.get(
        "/api/v1/announcements",
        headers={"Authorization": f"Bearer {company_token}"},
    )

    assert response.status_code == 200
    data = response.json
    assert "announcements" in data


def test_dismiss_announcement(client, company_token, sample_announcement):
    """Test de fermeture d'une annonce."""
    response = client.post(
        f"/api/v1/announcements/{sample_announcement.id}/dismiss",
        headers={"Authorization": f"Bearer {company_token}"},
    )

    assert response.status_code == 200


def test_is_visible_for_user():
    """Test de la logique de visibilité."""
    now = datetime.now(UTC)

    announcement = SystemAnnouncement(
        title="Test",
        message="Message",
        type="info",
        priority="normal",
        start_date=now,
        end_date=now + timedelta(days=1),
        is_active=True,
        is_published=True,
        target_roles=["company"],
    )

    # Visible pour entreprise
    assert announcement.is_visible_for_user("company")

    # Non visible pour chauffeur
    assert not announcement.is_visible_for_user("driver")
```

### Étape 9.2: Tests d'intégration Frontend

**Fichier**: `frontend/src/components/announcements/__tests__/AnnouncementBanner.test.jsx`

```javascript
// frontend/src/components/announcements/__tests__/AnnouncementBanner.test.jsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import axios from "axios";
import AnnouncementBanner from "../AnnouncementBanner";

jest.mock("axios");

describe("AnnouncementBanner", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should render announcements", async () => {
    axios.get.mockResolvedValue({
      data: {
        announcements: [
          {
            id: 1,
            title: "Test Announcement",
            message: "This is a test",
            type: "info",
            priority: "normal",
          },
        ],
      },
    });

    render(<AnnouncementBanner />);

    await waitFor(() => {
      expect(screen.getByText("Test Announcement")).toBeInTheDocument();
    });
  });

  it("should dismiss announcement on close", async () => {
    axios.get.mockResolvedValue({
      data: {
        announcements: [
          {
            id: 1,
            title: "Test Announcement",
            message: "This is a test",
            type: "info",
            priority: "normal",
          },
        ],
      },
    });

    axios.post.mockResolvedValue({ data: { message: "Dismissed" } });

    render(<AnnouncementBanner />);

    await waitFor(() => {
      expect(screen.getByText("Test Announcement")).toBeInTheDocument();
    });

    const closeButton = screen.getByLabelText("Fermer l'annonce");
    fireEvent.click(closeButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        "/api/v1/announcements/1/dismiss"
      );
    });
  });
});
```

---

## 🚀 Phase 10: Déploiement

### Étape 10.1: Déployer la migration

```bash
# Sur le serveur de production
ssh deploy@138.201.155.201

cd /srv/atmr

# Backup de la base de données (par sécurité)
docker compose -f docker-compose.production.yml exec -T postgres pg_dump -U atmr -d atmr > /srv/atmr/backups/pre-announcements-$(date +%Y%m%d-%H%M%S).sql

# Appliquer la migration
docker compose -f docker-compose.production.yml exec backend flask db upgrade

# Vérifier
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "\d system_announcements"
```

### Étape 10.2: Déployer le backend

```bash
# Build et redémarrage
cd /srv/atmr
git pull origin main

docker compose -f docker-compose.production.yml build backend
docker compose -f docker-compose.production.yml up -d backend

# Vérifier les logs
docker compose -f docker-compose.production.yml logs -f backend
```

### Étape 10.3: Déployer le frontend web

```bash
# Push vers Vercel (auto-déploiement)
git push origin main

# Vérifier le build sur Vercel dashboard
# https://vercel.com/votre-projet
```

### Étape 10.4: Déployer le mobile

```bash
# Build et submit mobile
cd mobile/operations-app

# iOS
eas build --platform ios --profile production
eas submit --platform ios

# Android
eas build --platform android --profile production
eas submit --platform android
```

### Étape 10.5: Tester en production

```bash
# 1. Créer une annonce test depuis le dashboard admin
# 2. Vérifier qu'elle apparaît sur les dashboards entreprise/chauffeur
# 3. Tester la fermeture
# 4. Créer une annonce critique et vérifier les push notifications
```

---

## 📊 Monitoring et maintenance

### Métriques à surveiller

1. **Performances API**:

   - Temps de réponse `/api/v1/announcements`
   - Nombre de requêtes par minute
   - Taux d'erreur

2. **Socket.IO**:

   - Nombre de connexions actives
   - Messages envoyés/reçus
   - Erreurs de connexion

3. **Push Notifications**:

   - Taux de livraison
   - Taux d'ouverture
   - Erreurs d'envoi

4. **Base de données**:
   - Nombre d'annonces actives
   - Nombre de vues
   - Nombre de fermetures

### Commandes utiles

```bash
# Compter les annonces actives
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "SELECT COUNT(*) FROM system_announcements WHERE is_active = true AND is_published = true;"

# Statistiques de vues
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "SELECT id, title, view_count, array_length(dismissed_by_user_ids, 1) as dismissals FROM system_announcements ORDER BY view_count DESC LIMIT 10;"

# Nettoyer les anciennes annonces (>6 mois)
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "DELETE FROM system_announcements WHERE end_date < NOW() - INTERVAL '6 months';"
```

---

## ✅ Checklist finale de déploiement

### Backend

- [ ] Migration appliquée en production
- [ ] API `/announcements` accessible
- [ ] Socket.IO diffuse correctement
- [ ] Push notifications fonctionnent
- [ ] Logs Sentry configurés
- [ ] Monitoring Grafana actif

### Frontend Web

- [ ] Bannière s'affiche correctement
- [ ] Socket.IO se connecte
- [ ] Fermeture fonctionne
- [ ] Responsive mobile OK
- [ ] Notifications navigateur OK

### Mobile

- [ ] Bannière s'affiche
- [ ] HTTP polling fonctionne
- [ ] Push notifications reçues
- [ ] Deep linking OK
- [ ] Badge count mis à jour

### Admin

- [ ] Interface de création OK
- [ ] Modification fonctionne
- [ ] Suppression fonctionne
- [ ] Statistiques affichées
- [ ] Permissions correctes

---

**Version**: 1.0  
**Dernière mise à jour**: 2026-01-13  
**Auteur**: Assistant IA  
**Status**: ✅ Prêt pour implémentation
