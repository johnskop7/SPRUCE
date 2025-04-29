## Architechture
lpips_type = 'vgg'
first_inv_type = 'w+'
optim_type = 'adamw'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 5
use_locality_regularization = True
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 10

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1.5

## Steps
LPIPS_value_threshold = 0.001
max_pti_steps = 1000
first_inv_steps = 450
max_images_to_invert = 500

## Optimization
pti_learning_rate = 1e-3
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = False
