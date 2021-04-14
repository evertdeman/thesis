# Identifying and predicting alcohol use in adolescents based on structural brain imaging

Master's thesis by Evert de Man. 

### Abstract 

Alcohol use disorder (AUD) destroys lives. Although many people consume alcohol, only some develop AUD. The identification of risk factors for AUD is of vital importance for the prevention, deceleration or treatment of AUD. Most people start experimenting with alcohol in adolescence and alcohol misuse during this period is associated with developing AUD later in life. One theory why adolescents are so vulnerable is that their developing brains (1) send out fewer cues to control alcohol intake and (2) might suffer more severe neurodegeneration due to alcohol use compared to adults. This degeneration may then further weaken control of intake, potentially creating a positive feedback loop that ends in AUD. So far, research about brain damage due to alcohol misuse in adolescents remains inconclusive.

This study aims to identify patterns in brain structure associated with binge drinking in adolescents in a multivariate pattern analysis (MVPA). Multiple machine learning models are trained on structural MRI (T1-weighted and DTI) data from the IMAGEN dataset to classify between 19-year-old binge drinkers and controls. Sex and scanner site confound the analyses. The study has two main takeaways. Firstly, it is shown that corrections for confounds in MVPAs should be performed with great care. We stress that researchers always explicitly report evidence that any confounding bias is correctly removed and we suggest how to do so. Secondly, none of the confound-corrected classifiers are able to classify binge drinkers from controls.

Explanations for this result might be that there is no (detectable) damage from binge drinking, that the damage is too heterogeneous among subjects or that the confound correction techniques not only destroy confounding signal, but also signal due to binge drinking. However, evidence is presented that argues against the latter. Repeating the analyses on a dataset with more severe adolescent drinkers might provide a clearer outcome. Moreover, classifying between different subtypes of adolescent drinkers (perhaps based on their consumption trajectory over time) could help to uncover the relationship between the adolescent's alcohol misuse, brain and increased vulnerability to developing AUD.


# Repository

This repository contains all the code that I used to get from the raw data in the IMAGEN dataset to the results presented in my thesis. 

The repository consists of the following folders: 
- `analysis` containing the machine learning analysis; 
- `inputcreator` containing a class to create HDF5 datasets from (preprocessed) data;
- `preprocessing` containing the code that I used to move raw whole-brain MRI images into BIDS format and subsequently preprocess (register to MNI space) them; 
- `vis` containing the code to visualize the results of the machine learning analysis.

Each folder contains its own README with relevant information. 

# Replicating my results 

All raw data is available in `/ritter/share/data/IMAGEN/IMAGEN_RAW/2.7/`, which is a mirrored version of the original IMAGEN server directory `/data/imagen/2.7/`.

## Preprocessing

I use three types of MRI data: 

1. Freesurfer statistics (`fs-stats`): this is already preprocessed by IMAGEN, no extra steps.
2. T1-weighted images (`T1w`): for this I used the `brainmask.mgz` image in the raw data. I moved those with `./preprocessing/bids.ipynb` into the BIDS-formatted folder `/ritter/share/data/IMAGEN/IMAGEN_freesurfer_BIDS/`. After this, I registered all scans with ANTs to MNI space with the script `./preprocessing/prep.py`. The preprocessed files are stored in `/ritter/share/data/IMAGEN/IMAGEN_prep-brainmask_BIDS/`. 
3. DTI fractional anisotropy (`dti-FA`): these are already preprocessed and registered to MNI space by IMAGEN. The only thing I did was move them into a BIDS-structured folder under `/ritter/share/data/IMAGEN/IMAGEN_DTI_BIDS/`.

## Creating HDF5 datasets

For all types of MRI, I use the `./inputcreator/inputcreator2.ipynb` to create HDF5 datasets that are ready to load into the classifier pipeline. HDF5 files are stored in `/ritter/share/data/IMAGEN/h5files/`. 

## Analysis

For the analysis I run `./analysis/analysis.py`.


# Features in the FSS data

For the description of all features included in the FSS data, please refer to the file `analysis/fnames.json`. 

