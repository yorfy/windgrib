@echo off
REM Script pour construire le package windgrib

echo Building windgrib package...

REM Activer l'environnement virtuel
call venv\Scripts\activate

REM Installer les outils de build si nécessaire
echo Installing build tools...
pip install build twine

REM Construire le package
echo Building package...
python -m build

REM Vérifier que les fichiers ont été créés
echo Checking build output...
if exist dist\windgrib-*.tar.gz (
    echo Package built successfully: dist\windgrib-*.tar.gz
) else (
    echo Error: Package build failed
    exit /b 1
)

if exist dist\windgrib-*.whl (
    echo Wheel built successfully: dist\windgrib-*.whl
) else (
    echo Error: Wheel build failed
    exit /b 1
)

echo Build completed successfully!
echo You can now upload to PyPI using: twine upload dist/*

pause