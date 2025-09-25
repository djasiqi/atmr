@echo off
setlocal
cd /d %~dp0

REM 1) lancer Docker Desktop si pas démarré
for /f "tokens=2,*" %%A in ('tasklist /fi "imagename eq Docker Desktop.exe" ^| find /i "Docker Desktop.exe"') do set RUNNING=1
if not defined RUNNING (
  echo [*] Lancement de Docker Desktop...
  start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  REM attendre que le daemon réponde
  set /a i=0
  :waitloop
  for /f "delims=" %%v in ('docker info 2^>nul ^| find "Server Version"') do set OK=1
  if not defined OK (
    set /a i+=1
    if %i% gtr 60 ( echo [!] Timeout attente Docker. Ouvrez Docker Desktop puis relancez. & exit /b 1 )
    ping -n 3 127.0.0.1 >nul
    goto waitloop
  )
)

echo [*] Contexte courant:
docker context ls

echo [*] Preparation des donnees OSRM (premiere fois ou si carte change)...
docker compose run --rm osrm_prepare || ( echo [!] Echec osrm_prepare & exit /b 1 )

echo [*] Demarrage du serveur OSRM...
docker compose up -d osrm || ( echo [!] Echec lancement osrm & exit /b 1 )

echo [*] Conteneurs:
docker ps

echo [*] Test endpoint:
curl "http://localhost:5001/table/v1/driving/6.1432,46.2044;6.1375,46.2100?annotations=duration"
echo.
echo [OK] OSRM pret sur http://localhost:5001
endlocal
