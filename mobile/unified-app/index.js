require("expo-asset");

const {
  registerDriverFcmBackgroundHandler,
} = require("./src/features/driver/firebaseMessaging");
const {
  registerDefaultSilentPushBackgroundHandler,
} = require("./src/features/driver/silentNotifications");
const {
  registerCompanyNotifeeBackgroundPressHandler,
} = require("./src/features/company/push/companyNotifeePress");

registerDefaultSilentPushBackgroundHandler();
registerDriverFcmBackgroundHandler();
registerCompanyNotifeeBackgroundPressHandler();

require("expo-router/entry");
