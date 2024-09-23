@echo off
REM Get the current date
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value ^| find "="')

REM Format the date
set "current_date=%datetime:~4,2%_%datetime:~6,2%"

start cmd /k "muselsl stream -p -c -g"

timeout /t 20 /nobreak

start cmd /k "muselsl record -d 30000 -dj True -t EEG"
start cmd /k "muselsl record -d 30000 -dj True -t PPG"
start cmd /k "muselsl record -d 30000 -dj True -t ACC"
start cmd /k "muselsl record -d 30000 -dj True -t GYRO"