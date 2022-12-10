#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

#Config variables
SERV_ADDR="0.0.0.0:9001"
SERVER_ROUNDS=3
STOP_PRUNE_ROUND=1
AGG_FRAC=1
BATCH_SIZE=64
PER_TRAIN_EPOCHS=3
PRUNING_FACTOR=1
LR_STEP_SIZE=20
TOT_CLIENTS=2
SAMPLE_CLIENTS=2
START_CLIENT_INDEX=0
END_CLIENT_INDEX=2
INIT_MODEL_FILE="../models/resnet18-round3.pth"
# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

echo "Starting server"
python server.py --serv_addr $SERV_ADDR --server_rounds $SERVER_ROUNDS --stop_prune $STOP_PRUNE_ROUND \
                    --init_model $INIT_MODEL_FILE --agg_fraction $AGG_FRAC\
                    --tot_clients $TOT_CLIENTS --sample_clients $SAMPLE_CLIENTS &
sleep 3 # Sleep for 3s to give the server enough time to start

for ((i=START_CLIENT_INDEX;i<END_CLIENT_INDEX;i++)); do
    echo "Starting client $i"
    python client.py --batch_size $BATCH_SIZE --total_epochs $PER_TRAIN_EPOCHS --step_size $LR_STEP_SIZE --prune_factor $PRUNING_FACTOR \
                        --client_index $i --num_clients $TOT_CLIENTS \
                        --serv_addr $SERV_ADDR &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
