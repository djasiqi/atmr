// src/components/layout/Sidebar/CompanySidebar/CompanySidebar.js
import React, { useEffect, useMemo, useState } from "react";
import { NavLink, useParams } from "react-router-dom";
import {
  FaHome,
  FaCar,
  FaUser,
  FaFileInvoice,
  FaCog,
  FaChevronRight,
  FaChevronLeft,
} from "react-icons/fa";
import styles from "./CompanySidebar.module.css";

const CompanySidebar = ({ isOpen, onToggle }) => {
  const { public_id } = useParams();
  const [open, setOpen] = useState(false);

  // Ajuste une variable CSS globale pour pousser le contenu
  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty("--sidebar-w", open ? "240px" : "72px");
    return () => root.style.removeProperty("--sidebar-w");
  }, [open]);

  const items = useMemo(
    () => [
      {
        to: `/dashboard/company/${public_id}`,
        label: "Tableau de bord",
        icon: <FaHome className={styles.icon} />,
        end: true,
      },
      {
        to: `/dashboard/company/${public_id}/reservations`,
        label: "Réservations",
        icon: <FaCar className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/drivers`,
        label: "Chauffeurs",
        icon: <FaUser className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/invoices/clients`,
        label: "Facturation par Client",
        icon: <FaFileInvoice className={styles.icon} />,
      },
      {
        to: `/dashboard/company/${public_id}/settings`,
        label: "Paramètres",
        icon: <FaCog className={styles.icon} />,
      },
    ],
    [public_id]
  );

  return (
    <aside
      className={`${styles.sidebar} ${open ? styles.open : styles.collapsed}`}
      aria-label="Navigation entreprise"
    >
      <button
        type="button"
        className={styles.toggler}
        aria-expanded={open}
        aria-label={open ? "Réduire le menu" : "Développer le menu"}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <FaChevronLeft /> : <FaChevronRight />}
      </button>


      <ul className={styles.menu}>
        {items.map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              end={!!item.end} // v6
              className={({ isActive }) => (isActive ? styles.active : undefined)} // v6
            >
              {item.icon}
              <span className={styles.text}>{item.label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </aside>
  );
};

export default CompanySidebar;
