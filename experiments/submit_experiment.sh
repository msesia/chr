#!/bin/bash

# Parameters
N_LIST=(100 150 200 250 300 400 500 1000 2000 3000 4000 5000)
SYMMETRY_LIST=(0 3 5 7 10 15 20 30) # 40 50
BATCH_LIST=$(seq 1 100)
BATCH_SIZE=1

# Slurm parameters
MEMO=5G                             # Memory required (5GB)
TIME=02:00:00                       # Time required (2h)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs"
mkdir -p $LOGS

OUT_DIR="results"
mkdir -p $OUT_DIR
mkdir -p "tmp_synthetic"

# Loop over configurations and chromosomes
for BATCH in $BATCH_LIST; do
  for N in "${N_LIST[@]}"; do
    for SYMMETRY in "${SYMMETRY_LIST[@]}"; do

      JOBN=$N"_"$SYMMETRY"_"$BATCH

      OUT_FILE=$OUT_DIR"/synthetic_s"$SYMMETRY"_n"$N"_b"$BATCH".txta"

      if [[ ! -f $OUT_FILE ]]; then
        # Script to be run
        SCRIPT="experiment_synthetic.sh $N $SYMMETRY $BATCH $BATCH_SIZE"
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
        ./$SCRIPT
      fi

    done
  done
done
