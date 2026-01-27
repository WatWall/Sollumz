@echo off
set "ZIP_NAME=sollumz_plus.zip"

echo ========================================
echo Building Sollumz+ Extension for Blender
echo ========================================

if exist %ZIP_NAME% del %ZIP_NAME%

echo Compressing files...
powershell -Command "$exclude = @('.git*', '.github', 'tests', 'prepare_release.ps1', 'build_extension.bat', '*.zip', '.vscode', '.agent', 'cliff.toml', 'requirements.txt', 'setup.cfg', 'pyproject.toml'); Get-ChildItem -Exclude $exclude | Compress-Archive -DestinationPath %ZIP_NAME% -Force"

if exist %ZIP_NAME% (
    echo.
    echo SUCCESS: %ZIP_NAME% created!
    echo.
    echo To install in Blender 4.2+:
    echo 1. Open Blender
    echo 2. Go to Edit > Preferences > Get Extensions
    echo 3. Click the arrow in the top right and select 'Install from Disk'
    echo 4. Select %ZIP_NAME%
) else (
    echo.
    echo ERROR: Failed to create %ZIP_NAME%
)

pause
