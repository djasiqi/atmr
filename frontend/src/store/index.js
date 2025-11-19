import { configureStore } from '@reduxjs/toolkit';
import userSlice from './slices/userSlice';
import authSlice from './slices/authSlice';
import reservationSlice from './slices/reservationSlice';

const store = configureStore({
  reducer: {
    user: userSlice,
    auth: authSlice,
    reservations: reservationSlice,
  },
});

export default store;
