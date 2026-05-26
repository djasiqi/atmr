import React, { useCallback, useEffect, useRef, useState } from 'react';
import Input from './Input';

/**
 * Champ texte local : la saisie ne re-render pas le parent.
 * Le parent lit la valeur via valueRef ; sync() pousse une mise à jour externe (ex. autocomplétion).
 */
export function useIsolatedField(initial = '') {
  const valueRef = useRef(initial);
  const [externalValue, setExternalValue] = useState(initial);

  const sync = useCallback((updater) => {
    const next =
      typeof updater === 'function' ? updater(valueRef.current) : (updater ?? '');
    valueRef.current = next;
    setExternalValue(next);
  }, []);

  return { valueRef, sync, externalValue };
}

export const IsolatedTextInput = React.memo(function IsolatedTextInput({
  externalValue = '',
  valueRef,
  onChange,
  ...props
}) {
  const [local, setLocal] = useState(externalValue);
  const lastExternalRef = useRef(externalValue);

  useEffect(() => {
    if (externalValue !== lastExternalRef.current) {
      lastExternalRef.current = externalValue;
      setLocal(externalValue);
      if (valueRef) valueRef.current = externalValue;
    }
  }, [externalValue, valueRef]);

  const handleChange = (e) => {
    const next = e.target.value;
    setLocal(next);
    if (valueRef) valueRef.current = next;
    onChange?.(e);
  };

  return <Input {...props} value={local} onChange={handleChange} />;
});

export const IsolatedTextarea = React.memo(function IsolatedTextarea({
  externalValue = '',
  valueRef,
  onChange,
  className = '',
  ...props
}) {
  const [local, setLocal] = useState(externalValue);
  const lastExternalRef = useRef(externalValue);

  useEffect(() => {
    if (externalValue !== lastExternalRef.current) {
      lastExternalRef.current = externalValue;
      setLocal(externalValue);
      if (valueRef) valueRef.current = externalValue;
    }
  }, [externalValue, valueRef]);

  const handleChange = (e) => {
    const next = e.target.value;
    setLocal(next);
    if (valueRef) valueRef.current = next;
    onChange?.(e);
  };

  return <textarea {...props} className={className} value={local} onChange={handleChange} />;
});
