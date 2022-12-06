# Cross validation splitters

Similarly to what is done in `scikit-learn` here we have implemented a number of cross-validation techniques, useful to **split** data into adequately formed subsets of train, test and validation data.

# Problems with cross validation for time series data

Random train-test splits can lead to data leakage, and if traditional k-fold and leave-one-out CV are the default
procedures being followed, data leakage will happen.
Leakage is the major reason why traditional CV is not appropriate for time series. Using these cross-validators
when you shouldn't will inflate performance metrics since they allow the training model to cheat because observations
"from the future" (posterior samples) leak into the training set.

Since time series data is time ordered, we want to keep intact the fact that we must use past observations to predict
future observations. The randomization in the standard cross-validation algorithm does not preserve the time ordering,
and we end up making predictions for some samples using a model trained on posterior samples. While this is not
immediately a huge problem for some applications, it becomes a critical one if the time series data is often strongly
correlated along the time axis. The randomization of traditional CV will make it likely that for each sample in the
validation set, numerous strongly correlated samples exist in the train set. This defeats the very purpose of having
a validation set: the model essentially “knows” about the validation set already, leading to inflated performance
metrics on the validation set in case of overfitting.

One solution is to use Walk-forward cross-validation (closest package implementation being Time Series Split
in sklearn), which restricts the full sample set differently for each split, but this suffers from the problem that,
near the split point, we may have training samples whose evaluation time is posterior to the prediction time of
validation samples.

Such overlapping samples are unlikely to be independent, leading to information leaking from the  train set into the
validation set. To deal with this, purging can be used.

## Purging
Purging involves dropping from the train set any sample whose evaluation time is posterior to the earliest prediction
time in the validation set. This ensures that predictions on the validation set are free of look-ahead bias.
But since walk-forward CV has other grievances, like its lack of focus on the most recent data during training fold
construction, the following is becoming more widely adopted for time series data:

## Combinatorial cross-validation
Consider we abandon the requirement that all the samples in the train set precede the samples in the validation set.
This is not as problematic as it may sound. The crucial point is to ensure that the samples in the validation set are
reasonably independent from the samples in the training set. If this condition is verified, the validation set
performance will still be a good proxy for the performance on new data.

Combinatorial K-fold cross-validation is similar to K-fold cross-validation, except that we take the validation set to
consists in j<K blocks of samples. We then have K choose j possible different splits. This allows us to create
easily a large number of splits, just by taking j=2 or 3, addressing two other problems not mentioned that purging
did not address.

It is however clear that we cannot use combinatorial K-fold cross-validation as it stands. We have to make sure that
the samples in the train set and in the validation set are independent. We already saw that purging helps reduce
their dependence. However, when there are train samples occurring after validation samples, this is not sufficient.

## Embargoing
We obviously also need to prevent the overlap of train and validation samples at the right end(s) of the validation set.
But simply dropping any train sample whose prediction time occurs before the latest evaluation time in the preceding
block of validation samples may not be sufficient. There may be correlations between the samples over longer
periods of time. In order to deal with such long range correlation, we can define an embargo period after each right
end of the validation set. If a train sample prediction time falls into the embargo period, we simply drop the sample
from the train set. The required embargo period has to be estimated from the problem and dataset at hand.
A nice feature of combinatorial cross-validation is also that as each block of samples appears the same number of
times in the validation set, we can group them (arbitrarily) into validation predictions over the full dataset (keeping
in mind that these predictions have been made by models trained on different train sets). This is very useful to
extract performance statistics over the whole dataset.
