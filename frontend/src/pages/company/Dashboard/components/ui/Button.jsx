import React from 'react';

const Button = ({ children, type = 'button', onClick, className = '', ...props }) => (
  <button type={type} onClick={onClick} className={`btn btn-primary ${className}`} {...props}>
    {children}
  </button>
);

export { Button };
