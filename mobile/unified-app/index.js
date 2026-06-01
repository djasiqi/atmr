const {
  registerDriverFcmBackgroundHandler,
} = require("./src/features/driver/firebaseMessaging");
const {
  registerDefaultSilentPushBackgroundHandler,
} = require("./src/features/driver/silentNotifications");

registerDefaultSilentPushBackgroundHandler();
registerDriverFcmBackgroundHandler();

require("expo-router/entry");
