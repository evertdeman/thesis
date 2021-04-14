# inputcreator 

With these files you can combine IMAGEN input data with a label from one of the `psytools` questionnaires in the IMAGEN dataset. 

For this you need
- preprocessed data; 
- a psytools questionnaire question that you want to base your label on. 

In the end it will produce a HDF5 file with 
- `X` neuroimaging data (either parcellated Freesurfer statistics or whole-brain T1/DTI files); 
- `y` the label from a Psytools questionnaire question; 
- `s` the sex of the subject;
- `c` the scanner site of the subject;
- `group` a unique group combination of sex and site of the subject;
- `subjid` the unique subject ID in the IMAGEN dataset.

The main way to do this is to use `inputcreator.ipynb` which used the class `InputCreator` defined in `inputcreator.py`. 

### Masks

In the directory `masks` there is an MNI mask used in `InputCreator` to keep non-brain voxels exactly zero.
