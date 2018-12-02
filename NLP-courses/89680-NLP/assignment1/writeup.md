# Assginment 1 writeup
----
Shahar Siegman 011862141

#### 1. Describe how you handled unknown words in hmm1.

In the HMM approach, encountering an unknown word poses a challenge since the algorithm depends on word-specific emission probabilites.
Naively we could assume equal probablities for all tags, or take the average tag probabilities in the dataset as our guess. These solutions may be workable, however taking advantage of word morphologies and patterns will give better results.

When the test (untagged) input is scanned, and a word is hit that was not in the original (training) dataset, its form is checked against a few patterns. The emissions vector for this word is the average of:
- An emission profile characterstic of rare words (this was deduced from the average tag frequencies of all the words in the original dataset that were encountered 6 times* or less)

- An emission profile characteristic for all words matching the pattern in the original dataset. For example, if the word "inflating" is encountered during tagging and it did not appear in the train, it is found to match against the pattern 'ends with _ing_'. It will receive the average emission profile of words in the train with the same suffix.
- For words that match more than one pattern eg. 'Hiked', the average emissions of all matched patterns will be used
- For words that match none of the patterns, the fallback used in the emission vector of all rare words in the train data. The threshold for rareness is appearing 6 times or less. Actually most words seen 6 times or less are seen 1 time.
- The list of patterns used:
    - ends with _tion_
    - consonant, vowel, consonant, e. Matches "love", "have", "give", "live" and also "lime", "dime" and "mine".
    - consonant, consonant, vowel, consonant, e. Matches "shave", "shake", "brake", "snake", "write" etc.
    - contains a dot as the 2nd, 3rd or 4th character e.g. "Dem.", "Ill.", "Mr."
    - contains a hyphen
    - suffix based patterns (with minimum total word length)
        - ends with _ed_ (minimum total length 4)
        - ends with _nal_ (7)
        - ends with _al_ (6)
        - ends with _ality_ (9)
        - ends with _ing_ (7)
        - ends with _ship_ (7) (e.g friendship, relationship)
        - ends with _ful_ (6) (e.g. colorful, playful)
        - ends with _fully_ (8) (e.g. respectfully, beautifully)

#### 2. Describe your pruning strategy in the viterbi hmm.
Siginificant speedup of the Viterbi in the HMM can be achieved by trimming paths of zero probability. This is due to the Emission Vector containing only one or two nonzero entries, as most words and character sequences are associated with just one or two possible parts-of-speech. It also helps if $q$'s underlying data structure allows querying $q(r|t,t')$ for all $t'$ in a single query since tag triplets are also sparse.
The speedup that can be achieved depends on the data but since the average number of nonzero entries in $e$ is usually much smaller than $T$ it is very signficant.
Unfortunately time constraints did not allow me to realize these optimizations for the HMM Viterbi. The MEMM Viterbi is optimized.

#### 3. Report your test scores when running the each tagger (hmm-greedy, hmm-viterbi, maxent-greedy, memm-viterbi) on each dataset. For the NER dataset, report token accuracy accuracy, as well as span precision, recall and F1.

- HMM greedy: 88.49%
- HMM Viterbi: 91.7%
- MEMM greedy: 92.47%
- MEMM Viterbi: 


#### 4. Is there a difference in behavior between the hmm and maxent taggers? discuss.
The MEMM has a few advantages:
- The generic scheme allows adding potentially useful features easily such as the next words, which is more complicated to add in the HMM
- It is less restricted to assigining same POS tags to known words, since it draws both on the specific word and other features. The emissions, as we implemented them, limit to assiging only encountered tag (this can be relaxed by a more sophisticate e-calculation scheme, but it requires specific consideration whereas the MEMM supports this more natively).
- The lambdas are an additional tunable parameter, which requires either deep knowledge of the task or familiarity with the algorithm to calibrate. The performance degradation due to miscalibrated lambdas is not known. 
- The phrasing of the problem as a conventional ML problem allows employing any ML classification tool, which means potential for performance improvements with little coding effort.

The main advantage that I see in the HMM is that its logic can be followed, and debugged. For example, a misclassification can be traced to low probabilities of specific tag triplets, missing examples of other usages of a word (e.g. "force" only encountered as a verb, even though it can be noun), and so forth. This allows more insightful iterations. 

In terms of relying on others' work, it can be achieve in both schemes. with HMM, it would mean explicitly merging $e$ and $q$ data that were obtained from other datasets with your own. With MEMM, it would mean that scores obtained from other classifiers, applied to your own data, can be used as a feature in your own ML scheme.

#### 5. Is there a difference in behavior between the datasets? discuss.
The performance on the NER dataset is poorer. this is probably due to two related facts:
1. sparser classifications (most tags are "O"-other) are always harder to learn
1. Both HMM and MEMM algorithms employ the tag sequential structure as classification cues. In the Named Entities case, no such structure can be learned. 

#### 6. What will you change in the hmm tagger to improve accuracy on the named entities data?
1. I would increase the Markovian order of the $q$ calculation, to support recognition of long names
1. I would change how the emissions are calculated, by relying much more on patterns (primarily capitalization) rather than trying to learn specific words. This is because name distribution is generally much more long-tailed than natural language word frequencies. The only exception is "closed" datasets. For example, a dataset may cover only S&P 500 companies, or only cities and countries of the world. In these cases, having a list of thousands or tens of thousands of tagged words may be enough to get good classifications.

#### 7. What will you change in the memm tagger to improve accuracy on the named entities data, on top of what you already did?
I would add more features.

#### 8. Why are span scores lower than accuracy scores?
Span scores are lower because in order to predict a span correctly you need to correctly classify all the words in and surrounding the span. This makes the scoring scheme harsher than single-word accuracy score.


