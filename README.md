# Ultra-Low-dose Amyloid PET reconstruction by GAN with perceptual loss
@article{ouyang2019ultra,
  title={Ultra-low-dose PET Reconstruction using Generative Adversarial Network with Feature Matching and Task-Specific Perceptual Loss},
  author={Ouyang, Jiahong and Chen, Kevin T and Gong, Enhao and Pauly, John and Zaharchuk, Greg},
  journal={Medical physics},
  year={2019},
  publisher={Wiley Online Library}
}

## Packages and Tools
FSL, FreeSurfer, tensorflow, pydicom, nibabel, skimage, numpy, scipy

## Pre-processing
Run fs_reslice.sh to reslice each PET volume to 256x256x89
Run fsl_coreg_longo.sh to co-register T2, T2-FLAIR to T1, then all to PET
Run data_preparation.py to generate data for training and testing

## Training and Testing
### Basic GAN model
Run main.py to train the basic GAN model (change phase to test if testing)
python main.py --phase train --task petonly --is_gan --is_l1 --feat_match --g_times 2

### Task GAN model
Run data_preparation_classification.py to generate data for task-specific network (amyloid classifier)
Run amyloid_pos_neg_classification_main.py to get the pre-trained model for the classifier

Run main.py to train the task GAN model
python main.py --phase train --task petonly --is_gan --is_l1 --is_ls --is_lc --feat_match --g_times 2
