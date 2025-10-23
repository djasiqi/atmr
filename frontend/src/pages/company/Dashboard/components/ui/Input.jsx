import React from 'react';

const Input = React.forwardRef(({ className = '', ...props }, ref) => (
  <input ref={ref} className={`form-input ${className}`} {...props} />
));

export { Input };
