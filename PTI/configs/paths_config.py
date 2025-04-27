## Pretrained models paths
e4e = '/home/ubuntu/e4e_chest_xrays.pt'
stylegan2_ada = '/home/ubuntu/gan_chest_xrays.pkl'


## Dirs for output files
checkpoints_dir = '/home/ubuntu/PTI/checkpoints/dementia_moderate'
embedding_base_dir = '/home/ubuntu/PTI/embeddings'

## Input info
### Input dir, where the images reside
input_data_path = '/home/ubuntu/test_images_cardiomegaly'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'dementia_moderate'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

