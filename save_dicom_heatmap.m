clear
% addpath /usr/local/spm12
key_set = [1350, 1355, 1375, 1726, 1732, 1750, 1758, 1762, 1785, 1789, ...
          1791, 1816, 1827, 1838, 1905, 1907, 1923, 1947, 1961, 1965, ...
          1978, 2014, 2016, 2063, 2152, 2157, 2185, 2214, 2304, 2314, ...
          2317, 2376, 2414, 2416, 2425, 2427, 2482, 2511, 2516, 50767];
value_set = {'Patient01', 'Patient02', 'Patient03','Patient04','Patient05', ...
            'Patient06', 'Patient07', 'Patient08','Patient09','Patient10', ...
            'Patient11', 'Patient12', 'Patient13','Patient14','Patient15', ...
            'Patient16', 'Patient17', 'Patient18','Patient19','Patient20', ...
            'Patient21', 'Patient22', 'Patient23','Patient24','Patient25', ...
            'Patient26', 'Patient27', 'Patient28','Patient29','Patient30', ...
            'Patient31', 'Patient32', 'Patient33','Patient34','Patient35', ...
            'Patient36', 'Patient37', 'Patient38','Patient39','Patient40'};
M = containers.Map(key_set,value_set);

% test_set = [1375];
% test_set = [1355, 1732, 1947, 2516, 2063, 50767, 1375, 1758, 1923, 2425];
test_set = [1350, 1726, 1750, 1762, 1785, 1791, 1827, 1838, 1905, 1907,...
            1978, 2014, 2016, 2157, 2214, 2304, 2317, 2376, 2427, 2414,...
            1961, 2185, 2152, 1789, 1816, 1965, 2314, 2511, 2416, 2482];
nifti_path = '/home/jiahong/project_lowdose/nifti/petmr_9block_l1+lc+ls/';
save_path = '/home/jiahong/project_lowdose/dicom_petmr/';
mkdir(save_path);
fulldose_path = '/data3/Amyloid/reading/';
for i = test_set
    fulldose_subj_path = [fulldose_path M(i) '/Full_Dose/anon_1.dcm'];
    fulldose_info = dicominfo(fulldose_subj_path);
    uid = fulldose_info.StudyInstanceUID;
    start_loc = fulldose_info.ImagePositionPatient;
    PatientName.FamilyName = M(i);
    PatientName.GivenName = M(i);
    fname_fullfile = [nifti_path int2str(i) '.nii'];
    SeriesNo = floor(1000*rand());
    SeriesDisc = 'petmr_l1+lc+ls';
    save_subj_path = [save_path int2str(i) '/'];
    mkdir(save_subj_path);
    make_fake_dicom(save_subj_path, fname_fullfile, start_loc, PatientName, SeriesNo, SeriesDisc, uid);
end