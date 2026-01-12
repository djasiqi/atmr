@echo off
REM Script de redémarrage rapide du backend Flask pour Windows

echo ================================================
echo       REDEMARRAGE DU BACKEND FLASK ATMR
echo ================================================
echo.

cd /d "%~dp0"

echo [1/4] Verification des processus Flask en cours...
echo.

REM Arrêter les processus Flask existants
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [!] Des processus Python sont en cours. Voulez-vous les arreter ? [O/N]
    choice /C ON /N /M "Reponse : "
    if errorlevel 2 goto :skip_kill
    if errorlevel 1 (
        echo [*] Arret des processus Python...
        taskkill /F /IM python.exe /T >NUL 2>&1
        timeout /t 2 /nobreak >NUL
        echo [OK] Processus arretes
    )
) else (
    echo [i] Aucun processus Python actif detecte
)

:skip_kill
echo.
echo [2/4] Preparation de l'environnement...
echo.

REM Vérifier l'environnement virtuel
if exist "venv\Scripts\activate.bat" (
    echo [*] Activation de l'environnement virtuel...
    call venv\Scripts\activate.bat
    echo [OK] Environnement virtuel active
) else if exist "..\venv\Scripts\activate.bat" (
    echo [*] Activation de l'environnement virtuel...
    call ..\venv\Scripts\activate.bat
    echo [OK] Environnement virtuel active
) else (
    echo [!] Aucun environnement virtuel detecte
)

echo.
echo [3/4] Verification de la configuration...
echo.

REM Afficher la version Python
python --version

REM Vérifier si app.py existe
if not exist "app.py" (
    echo [ERREUR] Le fichier app.py n'existe pas dans ce repertoire!
    echo Repertoire actuel : %CD%
    pause
    exit /b 1
)

echo [OK] Fichier app.py trouve
echo.
echo [4/4] Demarrage du serveur Flask...
echo ================================================
echo.
echo [i] Pour arreter le serveur, appuyez sur Ctrl+C
echo.
echo ================================================
echo.

REM Démarrer Flask
python app.py

REM Si le script est arrêté
echo.
echo ================================================
echo       BACKEND ARRETE
echo ================================================
pause
