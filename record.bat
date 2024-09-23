@echo off

start cmd /k "muselsl stream"
start cmd /k "muselsl record -d 30000 -dj True -t EEG"
start cmd /k "muselsl record -d 30000 -dj True -t PPG"
start cmd /k "muselsl record -d 30000 -dj True -t ACC"
start cmd /k "muselsl record -d 30000 -dj True -t GYRO"