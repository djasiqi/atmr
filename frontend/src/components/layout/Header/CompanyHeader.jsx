// src/components/layout/Header/CompanyHeader.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Link, useParams, useLocation } from "react-router-dom";
import styles from "./CompanyHeader.module.css";

import useCompanyData from "../../../hooks/useCompanyData";
import useDispatchDelays from "../../../hooks/useDispatchDelays";
import { logoutUser } from "../../../utils/apiClient";

const API_BASE = (process.env.REACT_APP_API_BASE_URL || "").replace(/\/$/, "");

function getInitials(name = "") {
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() || "").join("") || "CO";
}

function resolveLogoUrl(url) {
  if (!url) return "";
  if (/^https?:\/\//i.test(url)) return url;

  // Si c'est un chemin d'upload, construire l'URL complète sans /api
  if (url.startsWith("/uploads/")) {
    const baseUrl = (process.env.REACT_APP_API_BASE_URL || "").replace(
      /\/api.*$/,
      ""
    );
    return `${baseUrl}${url}`;
  }

  return `${API_BASE}${url.startsWith("/") ? "" : "/"}${url}`;
}

const CompanyHeader = () => {
  const params = useParams();
  const location = useLocation();

  // Extraire public_id de manière stable depuis l'URL
  const routePublicId =
    params.public_id ||
    (() => {
      const match = location.pathname.match(/\/dashboard\/company\/([^/]+)/);
      return match ? match[1] : null;
    })();

  const companyData = useCompanyData() || {};
  const company = companyData.company || null;

  const [logoOk, setLogoOk] = useState(true);
  const name = company?.name || "Entreprise";

  const logoSrc = useMemo(() => {
    const abs = resolveLogoUrl(company?.logo_url);
    if (!abs) return "";
    // Ajouter un timestamp pour éviter le cache du navigateur
    const sep = abs.includes("?") ? "&" : "?";
    return `${abs}${sep}v=${Date.now()}`;
  }, [company?.logo_url]);

  // Reset logoOk quand l'URL change pour permettre un nouveau chargement
  useEffect(() => {
    setLogoOk(true);
  }, [company?.logo_url]);

  const homeHref = routePublicId
    ? `/dashboard/company/${routePublicId}`
    : "/dashboard/company";

  // 🆕 Hook pour les retards (refresh toutes les 2 minutes)
  const { delayCount, hasCriticalDelays } = useDispatchDelays(null, 120000);

  return (
    <header className={styles.header} role="banner">
      <Link
        to={homeHref}
        className={styles.brand}
        aria-label="Tableau de bord entreprise"
      >
        <div className={styles.logoWrap}>
          {logoSrc && logoOk ? (
            <img
              src={logoSrc}
              alt="Logo de l’entreprise"
              className={styles.logoImg}
              onError={() => setLogoOk(false)}
              loading="eager"
            />
          ) : (
            <div className={styles.logoFallback} aria-hidden="true">
              {getInitials(name)}
            </div>
          )}
        </div>
      </Link>

      <div className={styles.rightZone}>
        {/* 🆕 Badge de notification retards */}
        {delayCount > 0 && (
          <Link
            to={`/dashboard/company/${routePublicId}/dispatch/monitor`}
            className={`${styles.delayBadge} ${
              hasCriticalDelays ? styles.critical : ""
            }`}
            title={`${delayCount} retard(s) détecté(s) - Cliquer pour voir les détails`}
          >
            <span className={styles.badgeIcon}>
              {hasCriticalDelays ? "🚨" : "⚠️"}
            </span>
            <span className={styles.badgeCount}>{delayCount}</span>
          </Link>
        )}

        <button type="button" className={styles.logoutBtn} onClick={logoutUser}>
          Déconnexion
        </button>
      </div>
    </header>
  );
};

export default CompanyHeader;
