# Deep Learning methods - class notes
## lecture 1 15 Oct 18
There are several types of data in the world. images. tabular. measurements - usually a sequence over time. a signal that evolves with time. Things that are naturally a sequence. Related but not connected to 680. 

To effectively work in the subject you have to know both. Considered an advanced course. Yossi's ML course will make it easier. Now I will cover in 2 lectures something that you should know, if you don't, learn now.

The exercise will be a continuation of the lecture. There will be a TA (hopefully) that can take students questions.

The submission is must, it's not graded. 

The exam examines whether you know the material. the exercises are instructive.

4 programming exercises, similar to last years'. some involve math (not too high)

Machine learning - a plus

If you take the clayton book by me, and skip the language parts, you follow the course pretty well.

Lots of real-world cases are sequences. most courses are in image processing. If image processing is your thing, then this course is still a good promer.

[Graham Neweby course](http://www.phontron.com/class/nn4nlp2018/schedule.html) has about twice as much material (2 lectures a week)

### What is learning
I learn, then I get tested on what I learned. It has to be nontrivial (not just retrieval) but in the same subject matter.

The induction problem: I learned Male/Female (from faces in Israel) and I fail on a test of Chinese students.

### types of learning
- supervised - someone gives me examples (tagged data). From their I need to generalize
   - regression - predict something continuous. from traits of persons, need to predict life expectancy. features of apartments, how much they cost.
   - classificiation - one of a few classes. will this stock price rise or fall. will this person live to be 80? 90? face -> gender, client -> churn, face -> country.
   - structured חיזוי מבני - a few dependent questions. I have an image, draw a rectangle around the elephant. I could pose this as a classification problem. is this pixel part of the elephant? but my answers are dependent on each other. dependency between labels. I [YG] sometimes work on it.

- unsupervised: I get images, without knowing what they are. I need to separate them in a way that will make sense (images of dog in one group, cat in other group). The challenge is to find a metric that creates largest differences between groups and minimum differences within groups.

Everybody wants unsupervised, because it save the hassle of taggin. but it might mislead because the algo might converge to a question i did not intend it to solve.

- unsupervised - I have 1M images, only 1000 of which are labeled. I want to use the entire set to learn. I can say something like "I think these features signify a cat, let's check their frequency in the 1M set". I want to find a way for the 1M to help mee, even though they aren't labeled.

- reinforcement (which we might touch on towards the end of the course).
I don't have knowledge on what to do, but I get rewards. Such as game. a sequence of observations, actions and rewards. I need to understand which of my actions brought me to  agood place. its a desirable way to improve robotics: I want to pick an item, I have lots of control decisions in between, finally I know whether the something was picked up.


If I want to learn to distinguish apples from oranges, I need to represent the object as (numerical) features: weight, hardness, color, circumference,...

Maybe I will fail because I missed some important feature. Deep learning allows me to avoid manual feature engineering.

__Hypothesis class__: i need to decide the scope of my search. e.g. all 2nd degree functions.  Too wide class may hinder generalizaiton. 

__inductive bias__: How the system is biased because of the selection of hypothesis class.


### deep learning

![borel measurable](./1.png)

the problems related to sequences:

![kinds of problems](file:///C/Shahar/BarIlan/NLP-courses/89687-DL/2.png)

## learning

### housing data example
Classifying by a linear classifier. if $\hat w \cdot \hat x   + b  >0$ it belongs to one class, if <0, other class

### language classifier
I can take pairs of letters ("bigrams"). Count the instances of each pair so $\{x_i\}$ are the frequency of each pair. I will want weights $w^e_i$, which will score its chances of being English, and $w^g_i$ for German. My classifier will be based on comparing the scores $\sum{x_iw^e_i}$ and $\sum{x_iw^g_i}$

I can get this way, by thinking of $w$ as a matrix, to score each language (and I can score more languages than just these 2)

