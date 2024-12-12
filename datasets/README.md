# Train Flow Model and Generate Flow-synthetic Data

To get the ground truth nll and score (gradient of log density), we first train flow models on the real datasets, e.g. dino. 
Then we sample from pre-trained flow models to get flow-synthetic data, and use this data as training data for our diffusion model. Since flow models are fully reversable, we can get the ground truth nll and score of the flow-syntheic data. 

## Train Flows
```
python flow_synthetic_2Ddata.py --train_or_sample train --dataset XXX --output_dir XXXX
```

For example, train on dino dataset:
```train flows
python flow_synthetic_2Ddata.py --train_or_sample train --dataset dino --output_dir ./flow_synthetic_2d_checkpoints/
```

We offer the pretrained flow models, and they are stored in [./flow_synthetic_2d_checkpoints/](./flow_synthetic_2d_checkpoints/) folder. 

## Sample from Flow Models
We also offer flow-synthetic dataset in `flow_synthetic_2Ddata.py`. 

Here is an example call: 
`train_ds, val_ds, x_shape = sample_2d_synthetic(dataset)`

