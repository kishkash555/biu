## Machine Learning Exercise 2
Shahar Siegman
011862141

## Dataset analysis 
I performed a quick analysis of the dataset:
- plotting historgrams of one feature at a time, separated by the label
- calculating covariances between features
- Splitting by sex and plot histograms

#### Correlations between features
My ovservations based on this analysis were:
- The *length* and *diameter* features are highly correlated ($\rho \gt 0.98$)
- The above two features are also fairly correlated to the *height* feature ($\rho \approx 0.83$)
- The *whole_weight*, *shucked_weight*, *viscera_weight*, are all correlated with coefficients of $0.93$ or more.
- These weight features are correlated with the *shell_weight* feature with a coefficient above $0.87$.
- There is also signficant correlation (around $0.83$) between the weight and length features.
- age 0 abalones are smaller than abalones in age groups 1 and 2 (i.e. have lower values in the *length*, and *weight_* features), but the distributions have a significant overlap.
- age 1-2 abalones have very similar distributions for all features examined.


#### Feature distribution shift
For the classifier input, I decided to shift each feature by a fixed amount, so that its mean is roughly zero. This is *not* expected to improve the results since the bias term should adapt either way. The amounts of shift are the (rough) means in the training set.  

| Feature | shift amount |
|---------|-------------:|
| length | -0.5 |
| diameter | -0.4 |
| height | -0.14 |
| whole_weight | -0.8 |
| shucked_weight | -0.34 |
| viscera_weight | -0.17 |
| shell_weight | -0.23 |
#### Distribution scaling
I did not use scaling on the data.

#### Analysis summary
Based on this analysis, and assuming equal distribution between classes in the test set, I estimated that a linear classifier could reach an accuracy of 60%-65%. This is based on identifying correctly age-0 abalones plus doing a little better than blind guess when deciding between age groups 1 and 2.  

## Addition configuration details
#### Diminishing learning rate
My initial results had large variance. The final score after training the classifiers ranged from 40% to about 60%, depdending on the randomized shuffle order. These were the result regardless of how many epochs it ran (I tried 1-20). This issue affected all three classifiers. This was *not* a bug in my code, and it took me a few attempts until I found a fix:
- At first I tried adding features (e.g. squares of existing features), this did not help.
- Then I tried reducing the learning rate after each sample (using *inverse-time* learning-rate rule), this improved results but only very slighly. 
- Then I decided to use exponential decay, by halving the step size after each *epoch*. This works well for all three classifiers and it solved the issue. Accuracies are around 65% in all three classifiers in a 5-fold cross-validation scheme. This accuracy is reached after 15-20 epochs.

#### SVM $\lambda$
I tested different values of the SVM decay parameter $\lambda$. Small values (1 and below) were all OK without any perceptable differences in performance. Higher values had a slight negative effect. I used $0.1$ in the final code.



