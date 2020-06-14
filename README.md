# TestingML_Pipelines
My notes on software testing  ML pipelines


Some machine learning applications are intended
to learn properties of data sets where the correct
answers are not already known to human users. It is
challenging to test such ML software, because there is
no reliable test oracle. We describe a software testing
approach aimed at addressing this problem. 


ML Programs : "Programs which were written in
order to determine the answer in the first place. There
would be no need to write such programs, if the
correct answer were known” 


## Challenges and potential complications
 - Huge volumes of collected data present storage and analytics challenges — scrubbing this amount of data can be incredibly time-consuming.
 - Data may be collected during unanticipated events or circumstances, making it difficult to gather and use for training purposes.
 - Human bias may appear in training and testing data sets.
 - Defects quickly fester and grow more complex in ML systems.


## Key aspects of testing
#### Data validation
> The key to successful AI/ML is good data. Before it’s supplied to an AI system, your data should be scrubbed, cleaned, and validated. Your QA team should be wary of human bias and variety that can complicate the system’s interpretation of the data — think of a car navigation system or smartphone assistant trying to interpret a rare accent.

#### Principle algorithms
> At the heart of AI/ML is the algorithm, which processes data and generates insights. Some common algorithms relate to learnability (the ability of Netflix or Amazon to learn customer preferences and serve new recommendations), voice recognition (smart speakers), and real-world sensor detection (self-driving cars). These should be tested thoroughly with model validation, successful learnability, algorithm effectiveness, and core understanding in mind. If there’s an issue with the algorithm, there are sure to be more serious consequences down the road.

#### Performance and security testing
> Just like any other software platform, AI systems require intensive performance and security testing, along with regulatory compliance testing. Without proper testing, niche security breaches (using voice recordings to fool voice recognition software or chatbot manipulation) will become more common.

#### Systems integration testing
> AI systems are built to hook into other systems and solve problems in a much larger context. For all of these integrations to work correctly, it’s necessary to perform a complete assessment of the AI system and its various connection points. With more and more systems absorbing AI characteristics, it’s vital that they’re tested carefully.


## Testing machine learning systems
> The goal of ML systems is to acquire knowledge on their own, without being explicitly programmed. This requires a consistent stream of data to be fed into the system — a much more dynamic approach that traditional testing is based on (fixed input = fixed output). Accordingly, QA experts will need to think differently about implementing test strategies for ML systems.

#### Training data and testing data
 > Training data is the set of data that is used to train the model for the system. In this data set, the input data is supplied along with the anticipated output. This is typically prepared by collecting data in a semi-automated way. Testing data is a subset of the training data, logically built to test all the possible combinations and determine how well your model is trained. Based on the results of the test data set, the model will be fine-tuned.

#### Model validation
> Test suites should be created to validate the system’s model. The principal algorithm analyzes all of the data provided, looks for specific patterns, and uses the results to develop optimal parameters for creating the model. From there, it is refined as the number of iterations and the richness of the data increases.

#### Communicating test results
> QA engineers are used to expressing the results of testing in terms of quality, such as defect leakage or the severity of defects. But the validation of models based on machine algorithms will produce approximations—not exact results. The engineers and stakeholders will need to determine the acceptable level of assurance, within a certain range for each outcome.




