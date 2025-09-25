import React from "react";
import styles from "./Input.module.css";

const Input = React.forwardRef(({ className = "", ...props }, ref) => (
  <input ref={ref} className={`${styles.input} ${className}`} {...props} />
));

export { Input };
