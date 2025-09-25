// src/components/layout/Header/CompanyHeader.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import styles from "./CompanyHeader.module.css";

import useCompanyData from "../../../hooks/useCompanyData";
import { logoutUser } from "../../../utils/apiClient";

const API_BASE = (process.env.REACT_APP_API_BASE_URL || "").replace(/\/$/, "");

function getInitials(name = "") {
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() || "").join("") || "CO";
}

function resolveLogoUrl(url) {
  if (!url) return "";
  if (/^https?:\/\//i.test(url)) return url;
  return `${API_BASE}${url.startsWith("/") ? "" : "/"}${url}`;
}

const CompanyHeader = () => {
  const { public_id: routePublicId } = useParams();

  const companyData = useCompanyData() || {};
  const company = companyData.company || null;

  const [logoOk, setLogoOk] = useState(true);
  const name = company?.name || "Entreprise";

  const logoSrc = useMemo(() => {
    const abs = resolveLogoUrl(company?.logo_url);
    if (!abs) return "";
    const sep = abs.includes("?") ? "&" : "?";
    return `${abs}${sep}v=${Date.now()}`;
  }, [company?.logo_url]);

  useEffect(() => setLogoOk(true), [logoSrc]);

  const homeHref = routePublicId
    ? `/dashboard/company/${routePublicId}`
    : "/dashboard/company";

  return (
    <header className={styles.header} role="banner">
      <Link to={homeHref} className={styles.brand} aria-label="Tableau de bord entreprise">
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
        <button type="button" className={styles.logoutBtn} onClick={logoutUser}>
          Déconnexion
        </button>
      </div>
    </header>
  );
};

export default CompanyHeader;
