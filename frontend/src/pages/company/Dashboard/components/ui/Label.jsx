import React from 'react';

const Label = ({ children, htmlFor, className = '', ...props }) => (
  <label htmlFor={htmlFor} className={`form-label ${className}`} {...props}>
    {children}
  </label>
);

export { Label };
