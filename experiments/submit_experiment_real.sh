#!/bin/bash

# Parameters
DATASET_NAME_LIST=("meps_19" "meps_20" "meps_21" "facebook_1" "facebook_2" "bio" "blog_data")
#DATASET_NAME_LIST=("facebook_1")
BBOX_NAME_LIST=("RF") # "RF")
EXP_ID_LIST=$(seq 1 100)

# Slurm parameters
MEMO=5G                             # Memory required (2GB)
TIME=12:00:00                       # Time required (2h)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs"
mkdir -p $LOGS

OUT_DIR="results_real"
mkdir -p $OUT_DIR
mkdir -p "tmp_real"

TARGET_LINES=10

# Loop over configurations and chromosomes
for DATASET_NAME in "${DATASET_NAME_LIST[@]}"; do
  for BBOX_NAME in "${BBOX_NAME_LIST[@]}"; do
    for EXP_ID in $EXP_ID_LIST; do

      JOBN=$DATASET_NAME"_"$BBOX_NAME"_"$EXP_ID

      OUT_FILE=$OUT_DIR"/dataset_"$DATASET_NAME"_bbox_"$BBOX_NAME"_exp_"$EXP_ID".txt"

      RUN=0
      if [[ ! -f $OUT_FILE ]]; then
        RUN=1
      fi
      if [[ -f $OUT_FILE ]]; then
        # Count lines
        NUM_LINES=$(wc -l < $OUT_FILE)
        if [ $NUM_LINES -lt $TARGET_LINES ]; then
          echo "Number of lines found: "$NUM_LINES
          RUN=1          
        fi
      fi

      if [[ $RUN == 1 ]]; then
        # Script to be run
        SCRIPT="experiment_real.sh $DATASET_NAME $BBOX_NAME $EXP_ID"

        # Define job name for this chromosome
        OUTF=$LOGS"/"$JOBN".out"
        ERRF=$LOGS"/"$JOBN".err"
        # Assemble slurm order for this job
        ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
        # Print order
        echo $SCRIPT
        # Submit order
        #$ORD
        # Run command now
        #./$SCRIPT
      fi

    done
  done
done
