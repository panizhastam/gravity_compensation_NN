# gravity_compensation_NN
This code maps the angular position of the upper-limb exoskeleton to the corresponding actuation gain needed for gravity compensation. The data was aquired using very slow sinusoidal inputs to replicate non existing dynamics during movements.

We are then going to run this data into our own model which we calculated using Euler Lagrange method in Matlab: nominal_model.m. Then we will compare the two models and the error is therefore learned from this MLP model. 

# Prerequisites

```bash
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U scikit-learn
conda install pandas
conda install matplotlib
```
