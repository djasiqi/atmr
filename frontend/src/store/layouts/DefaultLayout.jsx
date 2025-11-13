import React from "react";
import { Toaster } from "react-hot-toast";
import Header from "../../components/layout/Header/Header";
import Footer from "../../components/layout/Footer/Footer";

const DefaultLayout = ({ children }) => {
  return (
    <div style={styles.container}>
      <Toaster
        position="top-right"
        gutter={12}
        toastOptions={{
          duration: 5000,
          style: {
            fontSize: "14px",
            borderRadius: "12px",
            padding: "16px 18px",
            boxShadow:
              "0 18px 40px rgba(15, 23, 42, 0.12), 0 4px 16px rgba(15, 23, 42, 0.08)",
          },
        }}
      />
      <Header />
      <main style={styles.main}>{children}</main>
      <Footer />
    </div>
  );
};

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    minHeight: "100vh",
  },
  main: {
    flex: "1",
    padding: "20px",
  },
};

export default DefaultLayout;
