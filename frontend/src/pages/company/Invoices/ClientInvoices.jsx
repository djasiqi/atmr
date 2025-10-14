import React from "react";
import styles from "../Dashboard/CompanyDashboard.module.css";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import InvoicesRegistry from "./registry/InvoicesRegistry";

const ClientInvoices = () => {
  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          <InvoicesRegistry />
        </main>
      </div>
    </div>
  );
};

export default ClientInvoices;