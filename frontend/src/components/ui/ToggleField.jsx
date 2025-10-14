// frontend/src/components/ui/ToggleField.jsx
import React from "react";
import styles from "./ToggleField.module.css";

const ToggleField = ({
  label,
  name,
  value,
  onChange,
  hint,
  disabled = false,
}) => {
  return (
    <div className={styles.toggleField}>
      <div className={styles.toggleRow}>
        <div className={styles.toggleInfo}>
          <label htmlFor={name} className={styles.toggleLabel}>
            {label}
          </label>
          {hint && <p className={styles.toggleHint}>{hint}</p>}
        </div>

        <label className={styles.toggleSwitch}>
          <input
            id={name}
            type="checkbox"
            name={name}
            checked={value}
            onChange={onChange}
            disabled={disabled}
          />
          <span className={styles.toggleSlider}></span>
        </label>
      </div>
    </div>
  );
};

export default ToggleField;
