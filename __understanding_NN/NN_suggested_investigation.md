# Neural Network examination for design iterations and performance tuning

## Background
Despite the vast renewed interest in the past few years in application of Neural Network techniques, MLP in particular, to classification problems in many fields, practitioners still often face a lack of tools and methods to aid them in the design and validation of their MLP-based models. MLP's popularity and quick adoption trend are fueled in large part by their flexibility, versatility and minimalistic presentation. However, when seeking to make repeating, methodical use of MLPs on real world classification problem, or when experiencing difficulties obtaining satisfactory results on a particular problem by existing practices, many researchers and developers discover that few guidelines exist on how to critically examine, "debug" and fine-tune existing networks. In terms of theoretical rigor, many attempts have been made to establish theoretical claims regarding MLP "behaviors", but few, if any, gained any traction or widespread interest. 

So it seems that generic MLP research faces a "catch" or paradox which can be summarised as follows:
- Genral, rigorous, "first principles" theoretical observations are often too weak to have any tangiable impact on prevailing practices of those focused on the day-to-day of developing practical applications.
- When insights are gained, and presented within the context of a specific field of research, practicioners do not have the tools to understand how and whether the results can be carried over to their specific case, and are therefore understandably reluctant to experiment and test blindly the scores of new methods and potential improvements published each year.

In other words, the challenge here is not in the ability of dedicated individuals to arrive at new insights, as naturally happens in any other field of scientific interest, it is in the ability of the community to sift through, test and peer-review these incremental achievements in order to evolve and improve the accepted best practices.

## Guidelines for structure of work
The examination so far gives rise to the following guidelines for anyone wishihg to "hold the sticks at both ends" namely, help practictioners while "advancing the front" of MLP research at the same time:
- The reseracher must address a significant problem that is experienced widely by MLP practicioners in many fields, and has no good  solution or best practices.
- The researcher must be able to substantiate her claims theoretically, in order to ensure and convince the community in the generality of the results, but avoid being pulled too much into the maths of the problem.
- The researcher must work her recommendations into a clear, concise and easy to follow set of practical steps, and
- The researcher must accompany her recommendations with clear and easy to follow guidelines on how to test the usefulness of the new procedure, including the recommended test setup, the collection, interpretation and assessment of the test results specifically for the problem at hand.


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

Any breakthrough in answering one or more of the questions above, may have practical benefits both in making any critical discussion of a particular network's performance more focused and in enabling more effective, less time consuming MLP design procedures.


## Existing literature
Sifting through existing literature is likely to yield many interesting observations, insights and directions. A preliminary literature survey has identified relevant literature dating back to the mid-Eighties. The literature identified thus far can be roughly categorized as follows:

