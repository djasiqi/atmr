import React from "react";
import styles from "./Button.module.css";

const Button = ({ children, type = "button", onClick, className = "", ...props }) => (
  <button type={type} onClick={onClick} className={`${styles.button} ${className}`} {...props}>
    {children}
  </button>
);

export { Button };
