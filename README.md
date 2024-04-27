# CMIPS Project
Computational Methods Investigating Psychosocial Stressors (CMIPS).

Code used in the doctoral dissertation of Cory J. Cascalheira to predict psychosocial stressors from the social media of sexual and gender minority adults.

## Using The Code
All code is free to use with attribution to the author, Cory J. Cascalheira.

If you use the bot detection tactics (BDTs) in your work, please cite the corresponding paper: https://psyarxiv.com/gtp6z/

## Order of Script Exection

**Clean and Preprocess, Part I**
0. src/recruitment_sources.R
1. src/bdt_script.R
2. src/clean/clean_qualtrics.R
3. src/clean/clean_strain.R
4. src/clean/combine_dichotomize.R
5. src/extract/*
6. src/clean/combine_social_media.R

**Analyze, Part I**
7. src/analyze/describe_social_media.R

**Clean and Preprocess, Part II**
8. src/clean/clean_social_media.py
9. src/clean/clean_social_media.R

**Create Features / NLP-Generated Independent Variables**
10. src/create_features/create_features_*
11. src/create_features/create_features_dassp/*
12. src/create_features/synthesize_features.R

**Merge Features and Survey Data**
13. src/combine_features_scores.R

**Analyze, Part II**
14. src/analyze/describe_social_media.R
15. src/analyze/describe_participants.R
16. src/analyze/describe_measures.R
