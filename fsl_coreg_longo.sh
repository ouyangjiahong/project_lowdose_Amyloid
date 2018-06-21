#!/bin/bash

source /usr/share/fsl/5.0/etc/fslconf/fsl.sh

DST_DIR="/home/jiahong/project_lowdose/temp/"$1
echo $DST_DIR
mkdir $DST_DIR
cp -r /data3/Amyloid/temp/$1/mr_nifti_orig $DST_DIR
cp -r /data3/Amyloid/temp/$1/pet_nifti $DST_DIR

echo Getting co-registering params in T1 space for $1 ...
/usr/share/fsl/5.0/bin/flirt -in $DST_DIR/mr_nifti_orig/T2.nii.gz -ref $DST_DIR/mr_nifti_orig/T1.nii.gz -out $DST_DIR/T2_nifti -omat $DST_DIR/T2_nifti.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear

/usr/share/fsl/5.0/bin/flirt -in $DST_DIR/mr_nifti_orig/T2_FLAIR.nii.gz -ref $DST_DIR/mr_nifti_orig/T1.nii.gz -out $DST_DIR/T2_FLAIR_nifti -omat $DST_DIR/T2_FLAIR_nifti.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear

#/usr/share/fsl/5.0/flirt -in $DST_DIR/ASL_CBF_nifti/ASL_CBF.nii.gz -ref $DST_DIR/T1_nifti/T1.nii -out $DST_DIR/ASL_CBF_nifti -omat $DST_DIR/ASL_CBF_nifti.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear

/usr/share/fsl/5.0/bin/flirt -in $DST_DIR/pet_nifti/500_.nii -ref $DST_DIR/mr_nifti_orig/T1.nii.gz -out $DST_DIR/Full_Dose_nifti -omat $DST_DIR/Full_Dose_nifti.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear


echo Getting inverse transform params for $1 ...
/usr/share/fsl/5.0/bin/convert_xfm -omat $DST_DIR/Full_Dose_nifti_inv.mat -inverse $DST_DIR/Full_Dose_nifti.mat

/usr/share/fsl/5.0/bin/convert_xfm -omat $DST_DIR/T2_nifti_inv.mat -concat $DST_DIR/Full_Dose_nifti_inv.mat $DST_DIR/T2_nifti.mat

#/usr/share/fsl/5.0/convert_xfm -omat $DST_DIR/ASL_CBF_nifti_inv.mat -concat $DST_DIR/Full_Dose_nifti_inv.mat $DST_DIR/ASL_CBF_nifti.mat

/usr/share/fsl/5.0/bin/convert_xfm -omat $DST_DIR/T2_FLAIR_nifti_inv.mat -concat $DST_DIR/Full_Dose_nifti_inv.mat $DST_DIR/T2_FLAIR_nifti.mat


echo Applying transforms for $1 ...
/usr/share/fsl/5.0/bin/flirt -in $DST_DIR/mr_nifti_orig/T2_FLAIR.nii.gz -applyxfm -init $DST_DIR/T2_FLAIR_nifti_inv.mat -out $DST_DIR/T2_FLAIR_nifti_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/pet_nifti/500_.nii

/usr/share/fsl/5.0/bin/flirt -in $DST_DIR/mr_nifti_orig/T2.nii.gz -applyxfm -init $DST_DIR/T2_nifti_inv.mat -out $DST_DIR/T2_nifti_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/pet_nifti/500_.nii

#/usr/share/fsl/5.0/flirt -in $DST_DIR/ASL_CBF_nifti/ASL_CBF.nii.gz -applyxfm -init $DST_DIR/ASL_CBF_nifti_inv.mat -out $DST_DIR/ASL_CBF_nifti_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/pet_nifti/500_.nii

/usr/share/fsl/5.0/bin/flirt -in $DST_DIR/mr_nifti_orig/T1.nii.gz -applyxfm -init $DST_DIR/Full_Dose_nifti_inv.mat -out $DST_DIR/T1_nifti_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/pet_nifti/500_.nii

mkdir $DST_DIR/mr_nifti_orig
