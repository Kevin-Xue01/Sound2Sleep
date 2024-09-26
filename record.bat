@echo off

REM Get the current date and time
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value ^| find "="') do set datetime=%%i

REM Format the date and time (MM-DD_HH)
set "current_date=%datetime:~4,2%-%datetime:~6,2%_%datetime:~8,2%"

REM Prompt the user for input
set /p user_input="Please enter name of subject: "

REM Display the input back to the user
echo You entered: %user_input%

REM start cmd /k "muselsl stream -p -c -g"
start cmd /k "muselsl stream -c -g"

timeout /t 25 /nobreak

REM Create file names for each recording type
set "eeg_file=data/%user_input%/EEG_%current_date%.csv"
REM set "ppg_file=data/%user_input%/PPG_%current_date%.csv"
set "acc_file=data/%user_input%/ACC_%current_date%.csv"
set "gyro_file=data/%user_input%/GYRO_%current_date%.csv"

start cmd /k "muselsl record -d 30000 -dj True -t EEG -f %eeg_file%"
REM start cmd /k "muselsl record -d 30000 -dj True -t PPG -f %ppg_file%"
start cmd /k "muselsl record -d 30000 -dj True -t ACC -f %acc_file%"
start cmd /k "muselsl record -d 30000 -dj True -t GYRO -f %gyro_file%"