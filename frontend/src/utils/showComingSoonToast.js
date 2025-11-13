import toast from "react-hot-toast";

const COMING_SOON_TOAST_ID = "global-coming-soon";

const cardBaseStyle = {
  display: "flex",
  alignItems: "flex-start",
  background: "#0f172a",
  border: "1px solid rgba(148, 163, 184, 0.3)",
  borderRadius: "20px",
  padding: "16px",
  gap: "16px",
  maxWidth: "360px",
  width: "calc(100vw - 32px)",
  color: "#f8fafc",
  boxShadow: "0 15px 35px rgba(15, 23, 42, 0.35)",
  transition: "transform 0.2s ease, opacity 0.2s ease",
  backdropFilter: "blur(6px)",
};

const iconStyle = {
  fontSize: "28px",
  lineHeight: 1,
};

const bodyStyle = { flex: 1 };

const titleStyle = {
  fontWeight: 600,
  fontSize: "16px",
  margin: "0 0 4px 0",
  color: "#f8fafc",
};

const messageStyle = {
  margin: 0,
  fontSize: "14px",
  color: "rgba(255, 255, 255, 0.85)",
};

const linkStyle = {
  color: "#5eead4",
  textDecoration: "underline",
};

const closeStyle = {
  background: "none",
  border: "none",
  color: "#94a3b8",
  fontSize: "20px",
  cursor: "pointer",
  marginLeft: "8px",
  lineHeight: 1,
  transition: "color 0.2s ease",
};

export const showComingSoonToast = () => {
  toast.dismiss(COMING_SOON_TOAST_ID);
  toast.custom(
    (t) => (
      <div
        role="status"
        aria-live="polite"
        style={{
          ...cardBaseStyle,
          opacity: t.visible ? 1 : 0,
          transform: t.visible ? "translateY(0)" : "translateY(-8px)",
        }}
      >
        <div aria-hidden="true" style={iconStyle}>
          🚧
        </div>
        <div style={bodyStyle}>
          <strong style={titleStyle}>Bientôt disponible</strong>
          <p style={messageStyle}>
            Notre équipe finalise cette section. Écrivez-nous à{" "}
            <a href="mailto:info@lirie.ch" style={linkStyle}>
              info@lirie.ch
            </a>{" "}
            pour être informé du lancement.
          </p>
        </div>
        <button
          type="button"
          onClick={() => toast.dismiss(COMING_SOON_TOAST_ID)}
          aria-label="Fermer"
          style={closeStyle}
        >
          ×
        </button>
      </div>
    ),
    {
      id: COMING_SOON_TOAST_ID,
      duration: 5000,
      position: "top-right",
      style: {
        padding: 0,
        background: "transparent",
        boxShadow: "none",
      },
    }
  );
};

export default showComingSoonToast;

