import React, { useState } from 'react';
import styles from '../Dashboard/CompanyDashboard.module.css';
import InvoicesRegistry from './registry/InvoicesRegistry';
import DemoInteractiveGuide from '../../../components/demo/DemoInteractiveGuide';

const ClientInvoices = () => {
  const [showInvoicesMiniGuide, setShowInvoicesMiniGuide] = useState(() => {
    try {
      return window.sessionStorage.getItem('demo_invoices_mini') === '1';
    } catch {
      return false;
    }
  });

  return (
    <main className={styles.content} data-tour-id="invoices-page">
      {showInvoicesMiniGuide && (
        <DemoInteractiveGuide
          role="invoices-mini"
          onFinish={() => {
            setShowInvoicesMiniGuide(false);
            try {
              window.sessionStorage.removeItem('demo_invoices_mini');
            } catch {
              // ignore
            }
          }}
        />
      )}
      <div data-tour-id="invoice-registry">
        <InvoicesRegistry />
      </div>
    </main>
  );
};

export default ClientInvoices;
