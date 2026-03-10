@echo off
setlocal

REM Lance le frontend en mode demo (API demo sur :5100)
set "REACT_APP_API_BASE_URL=http://127.0.0.1:5100/api/v1"
set "REACT_APP_DEMO_MODE=true"
set "REACT_APP_API_URL=http://127.0.0.1:5100/api/v1"
set "REACT_APP_SOCKET_URL=http://127.0.0.1:5100/socket.io"

echo ============================================
echo Lancement frontend en mode DEMO
echo API: %REACT_APP_API_BASE_URL%
echo ============================================

npm start
