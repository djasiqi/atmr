// Stub pour expo-secure-store sur le web
// Utilise localStorage comme alternative

const getItemAsync = async (key) => {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    console.warn('[SecureStore] localStorage.getItem failed:', error);
    return null;
  }
};

const setItemAsync = async (key, value) => {
  try {
    localStorage.setItem(key, value);
  } catch (error) {
    console.warn('[SecureStore] localStorage.setItem failed:', error);
    throw error;
  }
};

const deleteItemAsync = async (key) => {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.warn('[SecureStore] localStorage.removeItem failed:', error);
    throw error;
  }
};

module.exports = {
  getItemAsync,
  setItemAsync,
  deleteItemAsync,
};

