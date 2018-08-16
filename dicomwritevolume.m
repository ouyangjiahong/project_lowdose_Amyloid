function [] = dicomwritevolume(savepath, fname, start_loc, V, VS, UID, PatientName, SeriesNo, SeriesDisc)
%DICOMWRITEVOLUME   Write a volume as DICOM slices in a zip file.
%   DICOMWRITEVOLUME(FNAME, V) writes the 3-D volume data V to
%   the zip-file FNAME. The voxel size is assummed to be 1 mm
%   in each dimension. The UID is generated
%   randomly, the Patient Name is set as Anonymous, and the series number is set as 999.
%
%   DICOMWRITE(FNAME, V, VS) writes the 3-D volume data V to the
%   zip-file FNAME with the voxel spacing VS. VS is either a scalar
%   for isotropic volumes or a 3-dimensional array. The UID is generated
%   randomly, the Patient Name is set as Anonymous, and the series number is set as 999.
%
%   See also DICOMREADVOLUME, DICOMWRITE, DICOMREAD
%
%   Author: medical-image-processing.com
%   Modified by: Kevin Chen
%   Date: Mar/27/2018
%
%   DICOMWRITE(FNAME, V, VS, UID, PatientName, SeriesNo) writes the 3-D volume data V to the
%   zip-file FNAME with the voxel spacing VS, pre-set UID, Patient Name, and Series Number. VS is either a scalar
%   for isotropic volumes or a 3-dimensional array. 

if nargin<5
    VS = [1 1 1];
    UID = dicomuid;
    PatientName.FamilyName = 'Anonymous';
    PatientName.GivenName = 'Anonymous';
    SeriesNo = 999;
end

if nargin<6
    UID = dicomuid;
    PatientName.FamilyName = 'Anonymous';
    PatientName.GivenName = 'Anonymous';
    SeriesNo = 999;
end

if nargin<7
    PatientName.FamilyName = 'Anonymous';
    PatientName.GivenName = 'Anonymous';
    SeriesNo = 999;
end

if nargin<8
    SeriesNo = 999;
end

if numel(VS)==1
   VS = [VS(1) VS(1) VS(1)]; 
end

if numel(VS) ~= 3
   error('VS needs to be either a scalar or 3-d vector.'); 
end

% create a dicom header with the relevant information
info.SliceThickness = VS(3);
info.ImagerPixelSpacing = VS(1:2);
info.PixelSpacing = VS(1:2);
info.Width = size(V,1);
info.Height = size(V,2);
info.ColorType = 'grayscale';
info.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.128'; 
info.TransferSyntaxUID = '1.2.840.10008.1.2.1'; % Explicit VR Little Endian
info.SOPClassUID = '1.2.840.10008.5.1.4.1.1.128'; 
info.PhotometricInterpretation = 'MONOCHROME2';
info.PixelRepresentation = 1;
%info.WindowCenter = 0;
%info.WindowWidth = 1000;
info.RescaleIntercept = 0;
info.RescaleSlope = 1;
%info.RescaleType = 'HU';
info.LargestImagePixelValue = 32767;
info.SmallestImagePixelValue = 0;
info.Modality = 'PT';
info.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.3.100.7.1';
info.ReferencedSOPInstanceUID = UID;
info.StudyInstanceUID = UID;
info.SeriesInstanceUID = UID;
info.FrameOfReferenceUID = UID;
info.SynchronizationFrameOfReferenceUID = UID;
info.SeriesNumber = SeriesNo;
info.SeriesDiscription = SeriesDisc;
info.UID = UID;
info.PatientPosition = 'HFS';
info.ImageOrientationPatient = [1;0;0;0;1;0];
info.PatientName = PatientName;
% create dicom slices
fnames = cell(0);

for i=1:size(V,3)
    I = rot90(V(:,:,size(V,3)-i+1));
    cfn = sprintf('img_%d.dcm', i);
    cfn = [savepath cfn];
%     info.ImagePositionPatient = [0 0 (size(V,3)-i)*VS(3)]
%     info.SliceLocation = (size(V,3)-i)*VS(3)

    info.ImagePositionPatient = [start_loc(1), start_loc(2), (start_loc(3)-(i-1)*VS(3))]
    info.SliceLocation = start_loc(3)-(i-1)*VS(3)
%     info.ImagePositionPatient = [start_loc(1), start_loc(2), (end_z+(i-1)*VS(3))]
%     info.SliceLocation = end_z+(i-1)*VS(3)
%     keyboard;
    dicomwrite(I, cfn, info, 'CreateMode', 'copy');
    fnames{i} = cfn;
end

% zip the dicom slices
zip(fname, fnames);

% finally cleanup all slices
%for i=1:numel(fnames)
%   delete(fnames{i}); 
%end