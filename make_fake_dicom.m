% Date: Mar/27/2018;
% Name: make_fake_dicom.m;
% Function to convert nifti to dicom. Calls dicomwritevolume
% Dependencies: SPM, MATLAB dicom toolbox
% Inputs: - fname_fullfile: absolute path of the .nii file you wish to
%           convert
%         - PatientName: structure with fields FamilyName and GivenName,
%           both strings
%         - SeriesNo: number for Series Number (used for randomizing image types)
% Author: Kevin Chen
% To use: make_fake_dicom(fname_fullfile, PatientName)

function make_fake_dicom(savepath, fname_fullfile, start_loc, PatientName, SeriesNo, SeriesDisc, UID)

% Get file name only
[~,fname, ~] = fileparts(fname_fullfile);

% Use file name as patient name if not passed in
if nargin<4
    PatientName.FamilyName = fname;
    PatientName.GivenName = fname;
    SeriesNo = 999;
    UID = dicomuid;
end

% Use 999 if no series no. passed in
if nargin<5
    SeriesNo = 999;
    UID = dicomuid;
end

% Generate random UID if none passed in
if nargin<6
    UID = dicomuid;
end

% Read nii
hdr = spm_vol(fname_fullfile);
% Convert data type of image matrix for dicom to recognize
% V = uint16(100*spm_read_vols(hdr));
V = uint16(spm_read_vols(hdr));

% Use header orientation matrix for voxel size
% VS = [hdr.mat(1,1) hdr.mat(2,2) hdr.mat(3,3)];
VS = [1.1719,1.1719,2.7800];

% Main function
dicomwritevolume(savepath, fname, start_loc, V, VS, UID, PatientName, SeriesNo, SeriesDisc)