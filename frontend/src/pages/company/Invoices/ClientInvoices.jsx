import React, { useState } from 'react';
import styles from '../Dashboard/CompanyDashboard.module.css';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
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
    <div className={styles.companyContainer} data-tour-id="invoices-page">
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content} data-tour-id="invoice-registry">
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
          <InvoicesRegistry />
        </main>
      </div>
    </div>
  );
};

export default ClientInvoices;
