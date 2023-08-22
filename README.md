Flower-based implementation of federated learning using ResNet18 on CIFAR-10. Implements an adaptive pruning strategy, which allows clients to select a degree of pruning based on resource limitations (RAM, network bandwidth etc.).

Done as part of a group project for the CS8803 Systems for Machine Learning, Fall 2022.

### Team members
Srihas Yarlagadda - syarlagadda37@gatech.edu  
Vidushi Vashisth - vvashishth3@gatech.edu  
Aniruddha Mysore - animysore@gatech.edu  
Aaditya Singh - asingh@gatech.edu  

> Georgia Institute of Technology

## Steps to run

1. Create a virtual environment using venv in the repo directory
    python -m venv sysmlvenv/

2. Activate the venv using source sysmlvenv/bin/activate

3. Install requirements using pip install -r requirements.txt

4. Cd into src. Run using ./run.sh (Spawns 1 server, 2 clients)

*Running the 8 Client Example*

1. Cd into examples/8clients (with the same env activated)

2. To see network log output, in a terminal window type the following-
sudo -E env PATH=$PATH python network.py --serv_addr 0.0.0.0:9001
The final network usage will be printed when this process is killed. Every packet transferred will be logged.

3. In separate terminals, execute the following scripts in order-
./run_server.sh
(Wait till the first two clients spawn)
./run_client1.sh
./run_client2.sh

Each script corresponds to a group of clients, whose output will appear in each terminal window. run_server.sh also shows the server output.
