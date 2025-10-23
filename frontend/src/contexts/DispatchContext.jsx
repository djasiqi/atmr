import React, { createContext, useContext, useReducer, useCallback } from 'react';

// Actions types
const ACTIONS = {
  SET_DATE: 'SET_DATE',
  SET_DISPATCHES: 'SET_DISPATCHES',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_DISPATCH_MODE: 'SET_DISPATCH_MODE',
  SET_DELAYS: 'SET_DELAYS',
  SET_SUMMARY: 'SET_SUMMARY',
  UPDATE_DISPATCH: 'UPDATE_DISPATCH',
  ADD_DISPATCH: 'ADD_DISPATCH',
  REMOVE_DISPATCH: 'REMOVE_DISPATCH',
  SET_OPTIMIZER_STATUS: 'SET_OPTIMIZER_STATUS',
};

// État initial
const initialState = {
  date: new Date().toISOString().split('T')[0],
  dispatches: [],
  delays: [],
  summary: null,
  dispatchMode: 'semi_auto',
  loading: false,
  error: null,
  optimizerStatus: null,
};

// Reducer
function dispatchReducer(state, action) {
  switch (action.type) {
    case ACTIONS.SET_DATE:
      return { ...state, date: action.payload };

    case ACTIONS.SET_DISPATCHES:
      return { ...state, dispatches: action.payload, loading: false, error: null };

    case ACTIONS.SET_LOADING:
      return { ...state, loading: action.payload };

    case ACTIONS.SET_ERROR:
      return { ...state, error: action.payload, loading: false };

    case ACTIONS.SET_DISPATCH_MODE:
      return { ...state, dispatchMode: action.payload };

    case ACTIONS.SET_DELAYS:
      return { ...state, delays: action.payload };

    case ACTIONS.SET_SUMMARY:
      return { ...state, summary: action.payload };

    case ACTIONS.UPDATE_DISPATCH:
      return {
        ...state,
        dispatches: state.dispatches.map((d) =>
          d.id === action.payload.id ? { ...d, ...action.payload.updates } : d
        ),
      };

    case ACTIONS.ADD_DISPATCH:
      return {
        ...state,
        dispatches: [...state.dispatches, action.payload],
      };

    case ACTIONS.REMOVE_DISPATCH:
      return {
        ...state,
        dispatches: state.dispatches.filter((d) => d.id !== action.payload),
      };

    case ACTIONS.SET_OPTIMIZER_STATUS:
      return { ...state, optimizerStatus: action.payload };

    default:
      return state;
  }
}

// Context
const DispatchContext = createContext(null);

// Provider
export const DispatchProvider = ({ children }) => {
  const [state, dispatch] = useReducer(dispatchReducer, initialState);

  // Actions
  const actions = {
    setDate: useCallback((date) => {
      dispatch({ type: ACTIONS.SET_DATE, payload: date });
    }, []),

    setDispatches: useCallback((dispatches) => {
      dispatch({ type: ACTIONS.SET_DISPATCHES, payload: dispatches });
    }, []),

    setLoading: useCallback((loading) => {
      dispatch({ type: ACTIONS.SET_LOADING, payload: loading });
    }, []),

    setError: useCallback((error) => {
      dispatch({ type: ACTIONS.SET_ERROR, payload: error });
    }, []),

    setDispatchMode: useCallback((mode) => {
      dispatch({ type: ACTIONS.SET_DISPATCH_MODE, payload: mode });
    }, []),

    setDelays: useCallback((delays) => {
      dispatch({ type: ACTIONS.SET_DELAYS, payload: delays });
    }, []),

    setSummary: useCallback((summary) => {
      dispatch({ type: ACTIONS.SET_SUMMARY, payload: summary });
    }, []),

    updateDispatch: useCallback((id, updates) => {
      dispatch({ type: ACTIONS.UPDATE_DISPATCH, payload: { id, updates } });
    }, []),

    addDispatch: useCallback((dispatchItem) => {
      dispatch({ type: ACTIONS.ADD_DISPATCH, payload: dispatchItem });
    }, []),

    removeDispatch: useCallback((id) => {
      dispatch({ type: ACTIONS.REMOVE_DISPATCH, payload: id });
    }, []),

    setOptimizerStatus: useCallback((status) => {
      dispatch({ type: ACTIONS.SET_OPTIMIZER_STATUS, payload: status });
    }, []),
  };

  const value = {
    state,
    actions,
  };

  return <DispatchContext.Provider value={value}>{children}</DispatchContext.Provider>;
};

// Hook personnalisé pour utiliser le context
export const useDispatchContext = () => {
  const context = useContext(DispatchContext);

  if (!context) {
    throw new Error('useDispatchContext must be used within a DispatchProvider');
  }

  return context;
};

export default DispatchContext;
