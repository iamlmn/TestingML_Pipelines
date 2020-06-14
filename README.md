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


When it comes to data products, a lot of the time there is a misconception that these cannot be put through automated testing. Although some parts of the pipeline can not go through traditional testing methodologies due to their experimental and stochastic nature, most of the pipeline can. In addition to this, the more unpredictable algorithms can be put through specialised validation processes.

Let’s take a look at traditional testing methodologies and how we can apply these to our data/ML pipelines.

Testing pyramid : 
Your standard simplified testing pyramid looks like this:

# PYRAMID PIC

This pyramid is a representation of the types of tests that you would write for an application. We start with a lot of Unit Tests, which test a single piece of functionality in isolation of others. Then we write Integration Tests which check whether bringing our isolated components together works as expected. Lastly we write UI or acceptance tests, which check that the application works as expected from the user’s perspective.

When it comes to data products, the pyramid is not so different. We have more or less the same levels.


# ML Test Pyramid

Note that the UI tests would still take place for the product, but this blog post focuses on tests most relevant to the data pipeline.

Let’s take a closer look at what each of these means in the context of Machine Learning, and with the help fo some sci-fi authors.

# What are Unit tests?
“It’s a system for testing your thoughts against the universe, and seeing whether they match” - Isaac Asimov.

Most of the code in a data pipeline consists of a data cleaning process. Each of the functions used to do data cleaning has a clear goal. Let’s say, for example, that one of the features that we have chosen for out model is the change of a value between the previous and current day

``` python
def add_difference(asimov_dataset):
    asimov_dataset['total_naughty_robots_previous_day'] =        
        asimov_dataset['total_naughty_robots'].shift(1)
 
    asimov_dataset['change_in_naughty_robots'] =    
        abs(asimov_dataset['total_naughty_robots_previous_day'] -
            asimov_dataset['total_naughty_robots'])
 
    return asimov_dataset[['total_naughty_robots', 'change_in_naughty_robots', 
        'robot_takeover_type']]
```
Here we know that for a given input we expect a certain output, therefore, we can test this with the following code:
``` python
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from unittest import TestCase
 
def test_change():
    asimov_dataset_input = pd.DataFrame({
        'total_naughty_robots': [1, 4, 5, 3],
        'robot_takeover_type': ['A', 'B', np.nan, 'A']
    })
 
    expected = pd.DataFrame({
        'total_naughty_robots': [1, 4, 5, 3],
        'change_in_naughty_robots': [np.nan, 3, 1, 2],
        'robot_takeover_type': ['A', 'B', np.nan, 'A']
    })
 
    result = add_difference(asimov_dataset_input)
 
    assert_frame_equal(expected, result)
```

For each piece of independent functionality, you would write a unit test, making sure that each part of the data transformation process has the expected effect on the data. For each piece of functionality you should also consider different scenarios (is there an if statement? then all conditionals should be tested). These would then be ran as part of your continuous integration (CI) pipeline on every commit.

In addition to checking that the code does what is intended, unit tests also give us a hand when debugging a problem. By adding a test that reproduces a newly discovered bug, we can ensure that the bug is fixed when we think that is fixed, and we can ensure that the bug does not happen again.

Lastly, these tests not only check that the code does what is intended, but also help us document the expectations that we had when creating the functionality.


# What is Integration testing?
Because “The unclouded eye was better, no matter what it saw.” Frank Herbert.

These tests aim to determine whether modules that have been developed separately work as expected when brought together. In terms of a data pipeline, these can check that:

The data cleaning process results in a dataset appropriate for the model
The model training can handle the data provided to it and outputs results (ensurign that code can be refactored in the future).

So if we take the unit tested function above and we add the following two functions:

``` python
def remove_nan_size(asimov_dataset):
    return asimov_dataset.dropna(subset=['robot_takeover_type'])
 
def clean_data(asimov_dataset):
    asimov_dataset_with_difference = add_difference(asimov_dataset)
    asimov_dataset_without_na = remove_nan_size(asimov_dataset_with_difference)
 
    return asimov_dataset_without_na
```

Then we can test that combining the functions inside clean_data will yield the expected result with the following code:

``` python
def test_cleanup():
    asimov_dataset_input = pd.DataFrame({
        'total_naughty_robots': [1, 4, 5, 3],
        'robot_takeover_type': ['A', 'B', np.nan, 'A']
    })
 
    expected = pd.DataFrame({
        'total_naughty_robots': [1, 4, 3],
        'change_in_naughty_robots': [np.nan, 3, 2],
        'robot_takeover_type': ['A', 'B', 'A']
    }).reset_index(drop=True)
 
    result = clean_data(asimov_dataset_input).reset_index(drop=True)
 
    assert_frame_equal(expected, result)
```

Now let’s say that the next thing we do is feed the above data to a logistic regression model.

``` python
from sklearn.linear_model import LogisticRegression
 
def get_reression_training_score(asimov_dataset, seed=9787):
    clean_set = clean_data(asimov_dataset).dropna()
 
    input_features = clean_set[['total_naughty_robots', 
        'change_in_naughty_robots']]
    labels = clean_set['robot_takeover_type']
 
    model = LogisticRegression(random_state=seed).fit(input_features, labels)
    return model.score(input_features, labels) * 100
```
Although we don’t know the expectation, we can ensure that we always result in the same value. It is useful for us to test this integration to ensure that:

The data is consumable by the model (a label exists for every input, the types of the data are accepted by the type of model chosen, etc)
We are able to refactor our code in the future, without breaking the end to end functionality.
We can ensure that the results are always the same by providing the same seed for the random generator. All major libraries allow you to set the seed (Tensorflow is a bit special, as it requires you to set the seed via numpy, so keep this in mind). The test could look as follows:
``` python
from numpy.testing import assert_equal
 
def test_regression_score():
    asimov_dataset_input = pd.DataFrame({
        'total_naughty_robots': [1, 4, 5, 3, 6, 5],
        'robot_takeover_type': ['A', 'B', np.nan, 'A', 'D', 'D']
    })
 
    result = get_reression_training_score(asimov_dataset_input, seed=1234)
    expected = 40.0
 
    assert_equal(result, 50.0)
```
There won’t be as many of these kinds of tests as unit tests, but they would still be part of your CI pipeline. You would use these to check the end to end functionality for a component and would therefore test more major scenarios.

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




