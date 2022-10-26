Flower-based implementation of federated learning using ResNet18 on CIFAR-10.

Install requirements using pip install -r requirements.txt
Run using ./run.sh (Spawns 1 server, 2 clients)

Current simplifications -
1. Only 1/10th of the trainig set is used to keep training times small on local machines. Can be edited in client.py -> get_dataloader(). The training data is also not shuffled for splitting (to be changed).
2. Model is trained from scratch instead of starting from a stored instance. Change incoming.

Client Parameters (can be changed using command line arguments to a client instance started using python client.py):
1. --batch_size: batch size for gradient descent
2. --total_epochs: epochs per round of FL at each client
3. --step_size: learning rate scheduler step size
4. --client_index: index of the client used to partition the training set
5. --num_clients: total number of FL clients, used for partitioning