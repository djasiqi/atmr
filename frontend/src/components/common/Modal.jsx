// src/components/common/Modal.jsx
import React from "react";
import styles from "./Modal.module.css";

const Modal = ({ children, onClose }) => {
  const handleClickOutside = (e) => {
    if (e.target.className === styles.modal) {
      onClose();
    }
  };

  return (
    <div className={styles.modal} onClick={handleClickOutside}>
      <div className={styles.modalContent}>
        {children}
      </div>
    </div>
  );
};

export default Modal;
