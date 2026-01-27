import { useCallback, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

const getInitialSearch = (params) => {
  return (params.get('search') || params.get('q') || '').trim();
};

const getShouldFocus = (params) => params.get('focusSearch') === '1';

const useUrlSearchSync = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [state, setState] = useState({
    initialSearch: '',
    shouldFocus: false,
    initialized: false,
  });
  const didInitRef = useRef(false);

  useEffect(() => {
    if (didInitRef.current) return;
    didInitRef.current = true;

    const initialSearch = getInitialSearch(searchParams);
    const shouldFocus = getShouldFocus(searchParams);

    setState({
      initialSearch,
      shouldFocus,
      initialized: true,
    });
  }, [searchParams]);

  const consumeFocusFlag = useCallback(() => {
    setState((prev) =>
      prev.shouldFocus ? { ...prev, shouldFocus: false } : prev
    );
    if (searchParams.get('focusSearch') === '1') {
      const nextParams = new URLSearchParams(searchParams);
      nextParams.delete('focusSearch');
      setSearchParams(nextParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  return {
    initialSearch: state.initialSearch,
    shouldFocus: state.shouldFocus,
    consumeFocus: consumeFocusFlag,
    initialized: state.initialized,
  };
};

export default useUrlSearchSync;
