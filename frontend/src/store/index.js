import { configureStore } from '@reduxjs/toolkit';
import userSlice from './slices/userSlice';

// Le runtime principal repose sur React Query.
// Ce store Redux est conservé uniquement pour les écrans/tests legacy
// et doit rester compilable sans dépendre de slices supprimées.
const noopReducer = (state = {}) => state;

const store = configureStore({
  reducer: {
    user: userSlice,
    auth: noopReducer,
    reservations: noopReducer,
  },
});

export default store;
