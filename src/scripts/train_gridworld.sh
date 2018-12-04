# Shell script to run training script on CodaLab.
# To upload code directory:
#     .../ppg> cl upload src


##### Experiment name #####
EXP_NAME="ppg2"


##### CodaLab arguments #####
CODALAB_ARGS="cl run"

# Name of bundle
CODALAB_ARGS="$CODALAB_ARGS --name $EXP_NAME"

# Docker image (default: codalab/default-cpu)
CODALAB_ARGS="$CODALAB_ARGS --request-docker-image nikhilxb/python3.7:1.0"

# Explicitly ask for a worker with at least one GPU
# CODALAB_ARGS="$CODALAB_ARGS --request-gpus 1"

# Control the amount of RAM the run needs
# CODALAB_ARGS="$CODALAB_ARGS --request-memory 5g"

# Bundle dependencies
CODALAB_ARGS="$CODALAB_ARGS :src"  # Entire parent code directory


##### Command to execute #####
CMD="PYTHONPATH=src/ python3.7 src/scripts/train_gridworld.py $EXP_NAME"
CMD="$CMD --experiments_dir experiments/"

# Pass the command-line arguments through to override the above
if [ -n "$1" ]; then
  CMD="$CMD $@"
fi


##### Run on CodaLab #####
FINAL_COMMAND="$CODALAB_ARGS '$CMD'"
echo $FINAL_COMMAND
exec bash -c "$FINAL_COMMAND"
