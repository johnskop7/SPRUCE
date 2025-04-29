## Pretrained models paths
e4e = '/home/ubuntu/encoder4editing/chexpert_experiment/checkpoints/best_model.pt'
stylegan2_ada = '/home/ubuntu/results/00021-chexpert_images-paper256-resumecustom/network-snapshot-019353.pkl'


## Dirs for output files
checkpoints_dir = '/home/ubuntu/checkpoints/chexpert_cardiomegaly'
embedding_base_dir = '/home/ubuntu/embeddings'

## Input info
### Input dir, where the images reside
input_data_path = '/home/ubuntu/test_images_chexpert_cardiomegaly'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'chexpert_cardiomegaly'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

