import React from 'react';
import { useSearchParams } from 'react-router-dom';
import InvoicesRegistry from './registry/InvoicesRegistry';
import CashRegisterFollowUp from '../Settings/tabs/CashRegisterFollowUp';
import SettlementIncidentsFollowUp from '../Settings/tabs/SettlementIncidentsFollowUp';
import styles from './CompanyBillingSpace.module.css';

const VIEWS = [
  { id: 'factures', label: 'Factures' },
  { id: 'reglements', label: 'Règlements' },
  { id: 'caisse', label: 'Caisse chauffeurs' },
  { id: 'signalements', label: 'Signalements règlement' },
];

const KNOWN = new Set(VIEWS.map((view) => view.id));

function currentView(value) {
  return KNOWN.has(value) ? value : 'factures';
}

/**
 * Espace Facturation. L'émission reste dans Factures, bouton Nouvelle facture.
 * Une vue inconnue, y compris l'ancien onglet À préparer, ouvre ce registre.
 */
export default function CompanyBillingSpace() {
  const [searchParams, setSearchParams] = useSearchParams();
  const view = currentView(searchParams.get('vue'));

  const select = (id) => {
    const next = new URLSearchParams(searchParams);
    if (id === 'factures') next.delete('vue');
    else next.set('vue', id);
    setSearchParams(next);
  };

  return (
    <div>
      <nav className={styles.nav} aria-label="Facturation">
        {VIEWS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={view === item.id ? styles.tabActive : styles.tab}
            aria-current={view === item.id ? 'page' : undefined}
            onClick={() => select(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {view === 'factures' ? (
        <div data-tour-id="invoice-registry">
          <InvoicesRegistry />
        </div>
      ) : null}

      {view === 'reglements' ? <CashRegisterFollowUp focus="reglements" /> : null}

      {view === 'caisse' ? <CashRegisterFollowUp /> : null}

      {view === 'signalements' ? <SettlementIncidentsFollowUp /> : null}
    </div>
  );
}
