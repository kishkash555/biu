# Neural Network examination for design iterations and performance tuning

## Background
Despite the vast renewed interest in the past few years in application of Neural Network techniques, MLP in particular, to classification problems in many fields, practitioners still often face a lack of tools and methods to aid them in the design and validation of their MLP-based models. MLP's popularity and quick adoption trend are fueled in large part by their flexibility, versatility and minimalistic presentation. However, when seeking to make repeating, methodical use of MLPs on real world classification problem, or when experiencing difficulties obtaining satisfactory results on a particular problem by existing practices, many researchers and developers discover that few guidelines exist on how to critically examine, "debug" and fine-tune existing networks. In terms of theoretical rigor, many attempts have been made to establish theoretical claims regarding MLP "behaviors", but few, if any, gained any traction or widespread interest. 

So it seems that generic MLP research faces a "catch" or paradox which can be summarised as follows:
- Genral, rigorous, "first principles" theoretical observations are often too weak to have any tangiable impact on prevailing practices of those focused on the day-to-day development of practical machinep-learning applications.
- Potentially useful, practical insights are typically presented within the context of a specific field of research. Practicioners as a whole do not have the tools to understand whether and how a prticular result can be carried over and implemented in their specific case, and are therefore understandably reluctant to blindly experiment and test the scores of new methods and potential improvements published each year.

In other words, the challenge here is not in the ability of dedicated individuals to arrive at new insights. It is in the ability of the community to sift through, test and peer-review these incremental achievements in order to evolve and improve the accepted best practices.

## Guidelines for structure of work
The situation as described suggests that researches that would be aware to the misgivnings of their predecessors can mitigate some of the issues and succeed in "holding the sticks at both ends" namely, help practictioners while at the same time "advancing the front" of MLP "basic" research:
- The reseracher must address a significant problem that is experienced widely by MLP practicioners in many fields, and has no accepted solution or best practices.
- The researcher must be able to substantiate her claims theoretically, in order to ensure and convince the community in the generality of the results, but avoid being pulled too much into the maths and theoretical aspects of the problem.
- The researcher must work her recommendations into a clear, concise and easy to follow set of practical steps, and
- The researcher must accompany her recommendations with clear and easy to follow guidelines on how to evaluate the new procedures - before, during and after their application.


## Overview of frequently arising questions during MLp design
- Network topology
    - interplay and tradeoff between network overall topology (number and size of layers, connectivity) and speed and quality of the convergence of the learning process (SGD and its variants).
    - being able to make more informed decisions on network topologies, for example based on qualities of the input (training) data that are easily computable.
    - useful upper and lower bounds for learning rates.
- Other network design parameters
    - deeper understanding of regularization, how and when it improves the results, inferring optimal regulating measures (penalty functions) based on assumptions on the input and output spaces.
    - insights on the implication of choice of activation function, especially double-bounded activation functions (tanh, logistic fucntion) vs. half bounded activation functions (ReLU)
- Interpretation, comparison and heuristic assessment of trained classifiers
    - more tools to compare trained classifier, firstly classifiers sharing the same topology, but also classifier with differring topologies.

Any breakthrough in answering one or more of the questions above, may have practical benefits both in making any critical discussion of a particular network's performance more focused and in enabling more effective, less time consuming MLP design and improvement procedures.


## Existing literature
Sifting through existing literature is likely to yield many interesting observations, insights and directions, scattered in differnet journals, using different vernacular etc. A preliminary literature survey has identified at least three waves of relevant literature, the first one dating to the mid-Eighties and the most recent one having started about 7 years ago. The literature identified thus far can be roughly categorized as follows:
- Geometric interpretation of MLP, as a way to analyze its "classification power"
- heuristic measures for (geometric) input-space classification complexity
- comparison of learning techniques

From these three categories, the first two are more relevant to the current study. Further literature survey will be undertaken.

## Proposed research
The main points of the proposed research are as follows:
1. Characterization of input spaces for 2- or few-class classification problems.
1. Characterization of the learning power of different MLP network topologies including examination of the contribution of each individual learning iteration to the final classifier
1. Analysis of various design and process parameters, such as step size scheme, depth vs. bredth, and regularization schemes and their impact on the convergence and the final classifier

As this outline covers a very wide range of topics, it is likely the actual research will need to focus on the most crucial (or most promising) directions, giving the rest of the issues a rudamentary treatment. 

All research will be based on real-world problems in the domain of NLP.


https://apps.dtic.mil/dtic/tr/fulltext/u2/a229035.pdf
https://ieeexplore.ieee.org/document/23880
https://www.computer.org/csdl/trans/tp/1992/06/i0686.pdf

https://universe-review.ca/I01-01-complexity.pdf

https://arxiv.org/abs/cs/0402020

