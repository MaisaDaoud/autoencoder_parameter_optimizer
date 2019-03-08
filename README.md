# Autoencoder_Parameter_Optimizer

_Autoencoder_parameter_optimizer_ is a tool for optimizing the autoencoders training hypermeters using Prticle swarm optimization (PSO) algorithm, the current version of the tool only optimizes the "learning_rate, epochs, batch_size, n_layers" and returns the best solution the maximizes the accuracy of the random forest classifer.

The tool creates  Representation and checkpoint directories for the generated represeantations and trained models respectively. However, The tool over writes the models and the representations during the optimizations process and  does not save the best model. The method developed her can be seen as a surrogate model  for expectsting the best training parameters for a given dataset.

In this tool, I used the Radia Basis Function based Autoencoder(RBFA) which was acceted recently for IEEE ECE 2019 conference. This new variation of the autoencoder relies on radial activation to the neuronse in the first layer of the networ, a link to the paper will be provided soon.

### To run the code:
git clone the project and run 

```
python3 -W ignore main.py --dataset_name="dataset_name" --train_dataset_file="train_dataset_file.csv" --iterations=100 --rep_length=250
```
