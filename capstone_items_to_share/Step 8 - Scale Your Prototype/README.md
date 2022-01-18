# Outline

1. 1.0-mic-extract_FRILL_embeddings

   > We finish extracting the FRILL embeddings from all the data. This set of features is the foundation of all subsequent feature enginering.

1. 2.0-mic-explore_ternary_multiclass

   > Using just the FRILL embeddings, I compared multiclass approaches with a variety of models. There was little to no difference between OvO, OvR, or unspecified ternary approaches. Maybe the plain ternary slightly outperformed the other two cases.

1. 3.0-mic-explore_neutral_binary

   > Still using just the FRILL embeddings, we continued to test a variety of models but on the neutral vs. non-neutral case. We generally obtain the best performance in this classification case. The hope was to possibly use a binary neutral classifier in conjunction with a binary positive vs. negative or a negative vs. non-negative classifier.

1. 4.0-mic-explore_positive-negative_binary

   > Along the same lines of #3 above, we tested a variety models on positive vs. negative binary classification.

1. 5.0-mic-explore_hybrid_multiclass

   > We considered using a neutral vs. non-neutral classifier that passes through the non-neutral predictions to a positive vs. negative classifier. The results were on par with those obtained in #2 above.

1. 6.0-mic-explore_resampling+top_ensemble

   > This notebook investigated resampling. Resampling appears to have a lot of potential to boost performance, especially for the binary classification cases. However, it's a really cumbersome process that is highly sensitive to any dimensionality reduction included in the pipeline. We also tried a stacked ensemble in this notebook of the better-performing classifiers, which did not outperform its base classifiers. Cross-validated accuracies are around 55% at this point.

1. 7.0-mic-explore_tree-based_models

   > We took a closer look at tree ensembles. They were training slowly so I wanted to eliminate the slowest and/or least performant of these from further consideration since they are similar algorithms.

1. 8.0-mic-explore_neg-neu_hybrid

   > Following the same motivation in #5 above, I revisited the idea of a two-tiered binary classifier. Instead of positive vs. negative disambiguation, I examined negative vs. non-negative disambiguation with the final non-negative samples defaulting to positive. This worked a little better than #5 above, probably because these are the two binary cases with the best scores. Class imbalance is present to a lesser degree in negative vs. non-negative compared to positive vs. negative as well. We're getting cross-validated balanced accuracy scores as high as 57% now; raw accuracy would probably be higher.

1. 9.0-mic-explore_dimensionality_reduction

   > Henceforth, we start iterating between model/pipeline evaluation and feature engineering. In one of these cycles, I noticed that linear discriminant analysis (LDA) was a slightly more performant dimensionality reduction technique. Visualizing these two components (for three classes) revealed three clusters in the data. However, 5-fold cross-validated balanced accuracies are still hovering around 57%. At this point, sklearn's multilayer perceptron appears to be the most performant, but just by a little. Later on, we will find that the two LDA components are central to feature engineering. We also investigated outlier removal with local outlier factor in this folder.

1. 10.0-mic-prepare_train-test_splits_on_full_data

   > I was running into out-of-memory issues and rerunning feature engineering/resampling/selection/outlier analysis etc. with each notebook was unnecessarily time consuming. Notebooks in this folder prepare features for pre-computed 5-fold train-test splits and saves them to disk. Whenever I discovered a promising feature during feature engineering, I would return to this folder to write a new notebook to prepare those features for each fold.

1. 11.0-mic-evaluate_untuned_performant_models

   > At this point, we haven't discovered the utility of the LDA components yet. Notebooks in this folder compare different models. Although we've started tracking different metrics, scores haven't improved very much yet.

1. 12.0-mic-evaluate_calibrated_untuned_ternary_classifiers

   > At this point, we've shelved the idea of a hybrid two-tiered classifier for simplicity and ease of development. In this folder, we take a look at calibration of the classifiers. Based on the notebooks in this folder as well in subsquent folders (e.g. #13), it looks like it could help, depending on the model, with maybe a slight edge for isotonic calibration over sigmoid calibration. We're still working with just the FRILL embeddings, so scores are still hovering at the plateau.

1. 13.0-mic-begin_tuning_models

   > Tuning helped to varying degrees depending on the model. At this point, I figured my scores were as good as they were gonna get with the FRILL embeddings.

1. 14.0-mic-evaluate_keras_MLP

   > At this point, the classifiers from the linear model were the best performing. I hoped to break the plateau with a fleshed out MLP rather than sklearn's default. I was not able to surpass the performance of the classical classifiers, but I stopped investigating this approach when I discovered the utility of the LDA components. If configured well, deep learning still holds potential.

1. 15.0-mic-compare_outlier_removal_techniques

   > Working with the FRILL embeddings, I tried some outlier removal techniques. They changed scores only a little, depending on the model, sometimes for the worse.

1. 16.0-mic-evaluate_LDA_features

   > Having discovered the potential of LDA components in visualization, notebooks in this folder investigate them further. The FRILL embeddings constitute 2,048 dimensions while the LDA components are just 2. Best accuracy scores were being achieved by a support vector classifier (SVC) with Gaussian kernel on the FRILL embeddings, not surpassing 64.3%, whereas a sigmoid-calibrated SVC attained just under 61.3% accuracy on two LDA components, almost just as good! Best AUROC so far was about 75.9% using logistic regression on the FRILL embeddings, whereas logistic regression achieved an AUROC of about 75.1% on just the two components. Evidently, there is good information in the LDA components. These two components would become a nexus between the FRILL embeddings and subsequent feature engineering. Henceforth, the primary classifier under consideration is Gaussian naive Bayes.

1. 17.0-mic-explore_one-class_techniques

   > Notebooks in this folder reflect initial forays into feature engineering based on the LDA components of the FRILL embeddings. In this stage, several per-class features were extracted based on a one-class multinomial classification approach. For instance, local outlier factor scores were introduced as one-class features. In particular, one-class SVCs with various kernels were used to extract per-class features from the LDA components as well as the original FRILL embeddings. SVCs were performing well but so slowly as to significantly hinder development speed; using SVCs as feature extractors allows us to retain their skill without costly training and predict times, just more time in preprocessing that can be saved to disk. We extracted more one-class scores and LDA pairs from these FRILL-based features as well. Treating feature pairs and trios as Cartesian coordinates, I also converted these to polar/spherical coordinates; the angle (theta/phi) coordinates were particularly informative.

1. 18.0-mic-evaluate_FRILL-based_features

   > These notebooks assess the FRILL-based features explored in #17 and extracted in #10. We focused on ExtraTrees as a benchmark and GaussianNB in particular, which was one of the most performant on the FRILL-based features. In this stage, we assesed feature importances with tree-ensembles as well. Although we extracted SVC features with polynomial kernels of many degrees, we found the 6th degree polynomial kernel to be most informative. We also found that the magnitude (rho) components of the polar/spherical coordinate pairs/trios were less informative. We found as well that the LDA components of all the polar/spherical features were important. With the new features, scores are rising to 62.4% accuracy and 76.3% AUROC. We've also begun tracking log loss to include a proper scoring metric.

1. 19.0-mic-extract_FRILL-based_features_from_full_data

   > I took a calculated risk here and made some tweaks to the previously cross-validated feature engineering process. Namely, I eliminated redundant scaling steps, discarded the rho coordinates of the polar/spherical features, and only used polynomial kernels of degree 5 and 6 (in addition to other SVC kernels) for feature engineering. While examining tree-based importances, I noticed that random forest was performing well based on out-of-bag estimates; on the other hand, a bagged Gaussian naive Bayes (NB) classifier performed best with just the final two LDA components of all the retained angle coordinates (also based on out-of-bag estimation). The modified feature engineering process yielded good out-of-bag accuracy scores 92.7% for random forest), but it hasn't been cross-validated.

1. 20.0-mic-tune_and_train_prototypes

   > I trained a total of 4 prototypes. The first is a plain GaussianNB. The second is a bagging GaussianNB. The third is a random forest. The fourth is a soft voting ensemble of an isotonically calibrated bagging GaussianNB and an isotonically calibrated random forest. I tuned hyperparameters for these based on out-of-bag performance and final AUROC and log loss (not cross validated); I fixed some hyperparameters intuitively based on previous cross-validated observations with other features. I also adapted earlier custom cross validation code for the calibration cross validation and for cross-validated scoring using 7 folds that avoid speaker leakage. Top 7-fold cross-validated scores were 92.5% for accuracy, 98.9% for AUROC, and 0.204 for log loss. Comparing these scores may suggest that random forest outperforms Gaussian NB, but these scores are optimistic since I tuned hyperparameters on the whole dataset, so there is implicit leakage in the process; maybe the random forest is more overfit than the GaussianNB. Moreover, there is underlying exposure between train and test splits from feauture extraction. Nonetheless, within those 7 folds, training and test are free of speaker leakage for evaluation. The holdout sets will be the better indicator. Post hoc, I added some models with RidgeClassifier.

1. 21.0-mic-prepare_holdout_datasets_for_evaluation

   > I found three new datasets that I didn't encounter when I first gathered data. In this stage, those datasets are prepared for evaluation using the same preprocessing and feature extraction steps we used for the full training dataset.

1. 22.0-mic-evaluate_prototypes_with_holdout_data
   > Results were disappointing, with accuracies capping around 59% and AUROC around 64%. The RF turned out to be a red herring. I went back and trained a RidgeClassifier, but only got a little bump in accuracy. I have a hunch that the *surprise* samples may induce a lot of noise. LogisticRegression could still be good, but I was worried about how long it took for the optimizer to converge.
   - best AUROC: 64.3% (plain_GNB)
   - best accuracy: 59.3% (voting_ensemble_gnb_ridge)
   - best log loss: 1.293 (bagging_GNB)

1. 23.0-mic-train_new_prototype_with_simplified_pipeline
  > It turns out the feature extractors were quite bulky, up to 5 GB in size. Following my intuition after having progressed thus far, I drastically simplified the featurization process. Feature extractors are now about 30 MB. Accuracy really took a hit, but log loss improved at least.
  - best AUROC: 64.1% (ridge)
  - best accuracy: 50.7% (bagging_GNB)
  - best log loss: 1.145 (voting_gnb_ridge)

1. 24.0-mic-train_new_prototype_with_simplified_pipeline_and_no_song_data
  > Since scores really took a hit, I decided to try one more round of tweaks. Following my intuition after having progressed thus far, I revised the featurization process.
  - best AUROC: 63.6% (ridge)
  - best accuracy: 53.5% (voting ensemble of GNBfull and LogReg; log loss 1.1098, AUROC 62.4%)
  - best log loss: 1.058 (bagging_GNB on full features)

1. 25.0-mic-train_new_prototypes_with_minimal_feature_processing
  > The feature extractors were a huge bottleneck in deployment, so I thought I'd simplify even more. I must've overdid it since the new scores are so bad they aren't worth reporting. At least it is validation that all that feature engineering effort wasn't in vain. I'll try just one more time with the poly kernel features removed since those feature extractors take the longest by far.

1. 26.0-mic-train_new_prototypes_with_simplified_pipeline_and_no_poly_scores
  > I still had to do something about the horribly slow processing times. The polynomial kernel features were the most cumbersome and I was able to drop them without losing all skill. Some scores are actually improved!
  - best AUROC: 63.2% (ridge)
  - best accuracy: 57.8% (voting ensemble of LogReg on 2 features and ridge on all FRILL-derived)
  - best log loss: 1.009 (voting ensemble of LogReg on 2 features and ridge on all FRILL-derived)
  > The chosen model for deployment was actually the stacked passthrough classifier; it combines the strengths of the feature extractors with a naive bayes, ridge regression, and logistic regression (accuracy: 50.7%, auroc: 62.5%, log loss: 1.043). We didn't choose the model with any of the top scores since the differentiating scores are really measures like balanced accuracy and geometric mean (given class imbalance). Log loss is almost equivalent for most models.

1. 27.0-mic-incorporate_holdout_data_and_retrain
  > You'd expect a better model with better data. The selected pipeline (all feature extractors, ridge + gnb -> logreg stack) was retrained on all available data including the holdout data. The assumption is that including the holdout data will improve generalizable performance.

# Ideas for redevelopment
* The FRILL embedding module is some kind of TensorFlow object. It may be possible to tune the final representation to focus on the affective components of vocal utterances rather than non-semantic features generally.
* The holdout data may be aggregated with the training data for more diversity and hopefully better performance on unseen data. To better assess generalizable performance while maximizing training data, it would be useful to develop a leave-one-out cross validation harness with folds divided by language or perhaps data source.
* Resampling and feature augmentation are computationally and/or storage expensive techniques, but there was evidence early on to suggest their efficacy on these data.
* The classification threshold at the end of the pipeline has not yet been assessed. Tuning this hyperparameter may yield better performance.
* Many of the feature extractors also have hyperparameters that may be tuned.


# Get the data

- [Unified Multilingual Dataset of Emotional Human Utterances](https://github.com/michen00/unified_multilingual_dataset_of_emotional_human_utterances)
- [emotiontts](https://github.com/emotiontts/emotiontts_open_db)
- [Thorsten_OGV_emotional](https://zenodo.org/record/5525023)
- [x4nth055_SER_custom](https://github.com/x4nth055/emotion-recognition-using-speech)
