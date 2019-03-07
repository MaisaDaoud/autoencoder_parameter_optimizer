# autoencoder_parameter_optimizer
Autoencoder_parameter_optimizer is a tool for optimizing the autoencoders training hypermeters using Prticle swarm optimization (PSO) algorithm , the current version of the tool only 
optimizes the "learning_rate, epochs, batch_size, n_layers" and returns the best solution the maximizes the accuracy of the
random forest classifer 
-To run the code:
--python3 -W ignore main.py --dataset_name="dataset_name" --train_dataset_file="train_dataset_file.csv" --iterations=100 --rep_length=250
