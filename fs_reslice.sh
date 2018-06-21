#!/bin/bash

#mri_convert -oni 320 -onj 320 -onk 256 PET_100.nii.gz PET_rsl_test.nii
#mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 PET_rsl_test.nii PET_rsl_test1.nii
#mri_convert -oni 256 -onj 256 -onk 89 PET_rsl_test1.nii PET_100_amy.nii

#mri_convert -oni 320 -onj 320 -onk 256 PET_025_coreg.nii.gz PET_rsl_test.nii
#mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 PET_rsl_test.nii PET_rsl_test1.nii
#mri_convert -oni 256 -onj 256 -onk 89 PET_rsl_test1.nii PET_025_coreg_amy.nii

#mri_convert -oni 320 -onj 320 -onk 256 PET_005_coreg.nii.gz PET_rsl_test.nii
#mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 PET_rsl_test.nii PET_rsl_test1.nii
#mri_convert -oni 256 -onj 256 -onk 89 PET_rsl_test1.nii PET_005_coreg_amy.nii

#mri_convert -oni 320 -onj 320 -onk 256 PET_002_coreg.nii.gz PET_rsl_test.nii
#mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 PET_rsl_test.nii PET_rsl_test1.nii
#mri_convert -oni 256 -onj 256 -onk 89 PET_rsl_test1.nii PET_002_coreg_amy.nii

#mri_convert -oni 320 -onj 320 -onk 256 PET_001_coreg.nii.gz PET_rsl_test.nii
#mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 PET_rsl_test.nii PET_rsl_test1.nii
#mri_convert -oni 256 -onj 256 -onk 89 PET_rsl_test1.nii PET_001_coreg_amy.nii

mri_convert -oni 320 -onj 320 -onk 256 T1_nifti_coreg.nii.gz T1_nifti_inv_rsl_test.nii
mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 T1_nifti_inv_rsl_test.nii T1_nifti_inv_rsl_test1.nii
mri_convert -oni 256 -onj 256 -onk 89 T1_nifti_inv_rsl_test1.nii T1_nifti_coreg_amy.nii

mri_convert -oni 320 -onj 320 -onk 256 T2_nifti_coreg.nii.gz T2_nifti_inv_rsl_test.nii
mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 T2_nifti_inv_rsl_test.nii T2_nifti_inv_rsl_test1.nii
mri_convert -oni 256 -onj 256 -onk 89 T2_nifti_inv_rsl_test1.nii T2_nifti_coreg_amy.nii

rm *rsl_test*.nii