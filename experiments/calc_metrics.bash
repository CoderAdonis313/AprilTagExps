#!/usr/bin/env bash

START=$(date +%s.%N)
BASE_PATH=""

if [ "$HOME" = "/eng/home/atembulk" ]; then
    BASE_PATH="/data/users/atembulk/dev/AprilTagExps/experiments"
else
    BASE_PATH="/home/abhinav/Dev/AprilTagExps/experiments"
fi

python "$BASE_PATH/tag_metrics.py" "$BASE_PATH/calibs/calib_camera_0__facing_back_1920.json" "$BASE_PATH/at_poses" 

END=$(date +%s.%N)
DURATION=$(awk "BEGIN {print $END - $START}")

echo "Time taken: $DURATION seconds"