# TestingML_Pipelines
My notes on software testing/ Monitoring  ML pipelines. 

> "All code is guilty, until proven innocent".

With the increasing popularity of Machine learning today, it is important to focus the testing aspect of ML application. Testing of any ML application will not be same as testing traditional software. It has become a debatable topic. Many literature categorised ML application as non-testable. However, many are now trying to make it testable and coming up many innovative approaches. All are very much technical and normal users without technical knowledge find difficulties. Therefore, here I have tried to explain the difference of testing traditional software and ML application in a simple terminology without going into any technical term.

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

# Testing pyramid : 
Your standard simplified testing pyramid looks like this:

## PYRAMID PIC

This pyramid is a representation of the types of tests that you would write for an application. We start with a lot of Unit Tests, which test a single piece of functionality in isolation of others. Then we write Integration Tests which check whether bringing our isolated components together works as expected. Lastly we write UI or acceptance tests, which check that the application works as expected from the user’s perspective.

When it comes to data products, the pyramid is not so different. We have more or less the same levels.


## ML Test Pyramid

Note that the UI tests would still take place for the product, but this blog post focuses on tests most relevant to the data pipeline.

Let’s take a closer look at what each of these means in the context of Machine Learning, and with the help fo some sci-fi authors.


### What is the difference between testing traditional Software testing and testing a ML application?

> In Traditional software we check out exactly the output is as compared to expected. Example the (1) case. We set expectation 11 and I’ll check output is 11 or not. Here the person knows just a rule about addition and that is why answer is 11. However, rest of the examples are testing of ML application, where output is not exactly what I expected, but all are correct and close to my expectation. Hence, in ML application we’ll not test how exact the output is, rather how close the output to correctness.
> We need to understand that while you are testing ML application, you are basically testing a software which itself learns, not just sequence of rules. With same data as input, you might get different output in two different run. Therefore, testing ML application requires an entirely different approach and the test team also needs to be elevated with ML skill. Eventually, to adopt ML application, we also need a different culture across organisation to sync our expectation. Even support engineer also needs this skill to handle any incident for ML application. Most important we need to come out from traditional mindset and embrace a new way of thinking. With this basic understanding, I like to encourage reader to read any articles which are available in internet about testing of traditional application vs testing ML application, they will understand better. I'm very much open to have any new idea, new thought on this topic from all of you.

### Functional Testing or E2E
 > Functional testing is also sometimes called E2E testing, or browser testing. They all refer to the same thing.
 > Functional testing is defined as the testing of complete functionality of some application. In practice with web apps, this means using some tool to automate a browser, which is then used to click around on the pages to test the application.
 >You might use a unit test to test an individual function and an integration test to check that two parts of the play nice. Functional tests are on a whole another level. While you can have hundreds of unit tests, you usually want to have only a small amount of functional tests. This is mainly because functional tests can be difficult to write and maintain due to their very high complexity. They also run very slowly, because they simulate real user interaction on a web page, so even page load times become a factor.
 > Because of all this, you shouldn’t try to make very fine grained functional tests. You don’t want to test a single function, despite the name “functional” perhaps hinting at it. Instead, functional tests 
 should be used for testing common user interactions. If you would manually test a certain flow of your app in a browser, such as registering an account, you could make that into a functional test.
 > While in unit and integration tests you would validate the results in code, functional test results should be validated the same way as you would validate it if you were a user of the page. Going with the registration example, you could validate it by checking that the browser is redirected to a “thanks for registering page”.
 > You should use functional tests if you have some repeated tests you do manually in the browser, but be careful to not make them too fine-grained, as they can easily become a nightmare to maintain. I know, because I’ve seen it happen many times.


### Unit tests
“It’s a system for testing your thoughts against the universe, and seeing whether they match” - Isaac Asimov.
##### what is unit testing?
> Unit testing is the practice of testing small pieces of code, typically individual functions, alone and isolated. If your test uses some external resource, like the network or a database, it’s not a unit test.
> Unit tests should be fairly simple to write. A unit tests should essentially just give the function that’s tested some inputs, and then check what the function outputs is correct. In practice this can vary, because if your code is poorly designed, writing unit tests can be difficult. Because of that, unit testing is the only testing method which also helps you write better code – Code that’s hard to unit test usually has poor design.
> In a sense, unit testing is the backbone. You can use unit tests to help design your code and keep it as a safety net when doing changes, and the same methods you use for unit testing are also applicable to the other types of testing. All the other test types are also constructed from similar pieces as unit tests, they are just more complex and less precise.
Unit tests are also great for preventing regressions – bugs that occur repeatedly. Many times there’s been a particularly troublesome piece of code which just keeps breaking no matter how many times I fix it. 
> By adding unit tests to check for those specific bugs, you can easily prevent situations like that. You can also use integration tests or functional tests for regression testing, but unit tests are much more useful because they are very specific, which makes it easy to pinpoint and then fix the problem.
 > When should you use unit testing? Ideally all the time, by applying test-driven development. A good set of unit tests do not only prevent bugs, but also improve your code design, and make sure you can later refactor your code without everything completely breaking apart.

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


### Integration testing
Because “The unclouded eye was better, no matter what it saw.” Frank Herbert.

###### what is integration testing?
> As the name suggests, in integration testing the idea is to test how parts of the system work together – the integration of the parts. Integration tests are similar to unit tests, but there’s one big difference: while unit tests are isolated from other components, integration tests are not. For example, a unit test for database access code would not talk to a real database, but an integration test would.
> Integration testing is mainly useful for situations where unit testing is not enough. Sometimes you need to have tests to verify that two separate systems – like a database and your app – work together correctly, and that calls for an integration test. As a result, when validating integration test results, you could for example validate a database related test by querying the database to check the database state is correct.
> Integration tests are often slower than unit tests because of the added complexity. They also might need some set up or configuration, such as the setting up of a test database. This makes writing and maintaining them harder than unit tests, so you should focus on unit tests unless you absolutely need an integration test.
> You should have fewer integration tests than unit tests. You should mainly use them if you need to test two separate systems together, or if a piece of code is too complex to unit test. But in the latter case, I would recommend fixing the code so it’s easy to unit test instead.
> Integration tests can usually be written with the same tools as unit tests.

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

##### When is Integration Testing performed?
 > Integration Testing is the second level of testing performed after Unit Testing and before System Testing.

##### Who performs Integration Testing?
 > Developers themselves or independent testers perform Integration Testing.

##### Approaches?
 - Big Bang is an approach to Integration Testing where all or most of the units are combined together and tested at one go. This approach is taken when the testing team receives the entire software in a bundle. So what is the difference between Big Bang Integration Testing and System Testing? Well, the former tests only the interactions between the units while the latter tests the entire system.
 - Top Down is an approach to Integration Testing where top-level units are tested first and lower level units are tested step by step after that. This approach is taken when top-down development approach is followed. Test Stubs are needed to simulate lower level units which may not be available during the initial phases.
 - Bottom Up is an approach to Integration Testing where bottom level units are tested first and upper-level units step by step after that. This approach is taken when bottom-up development approach is followed. Test Drivers are needed to simulate higher level units which may not be available during the initial phases.
 - Sandwich/Hybrid is an approach to Integration Testing which is a combination of Top Down and Bottom Up approaches.

##### Tips
 - Ensure that you have a proper Detail Design document where interactions between each unit are clearly defined. In fact, you will not be able to perform Integration Testing without this information.
 - Ensure that you have a robust Software Configuration Management system in place. Or else, you will have a tough time tracking the right version of each unit, especially if the number of units to be integrated is huge.
 - Make sure that each unit is unit tested before you start Integration Testing.
 - As far as possible, automate your tests, especially when you use the Top Down or Bottom Up approach, since regression testing is important each time you integrate a unit, and manual regression testing can be inefficient


## Challenges and potential complications
 - Huge volumes of collected data present storage and analytics challenges — scrubbing this amount of data can be incredibly time-consuming.
 - Data may be collected during unanticipated events or circumstances, making it difficult to gather and use for training purposes.
 - Human bias may appear in training and testing data sets.
 - Defects quickly fester and grow more complex in ML systems.


## Key aspects of testing ML pipelines/Apps.
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



General Terms used in QA
## Software Testing Types
SOFTWARE TESTING TYPES listed here are a few out of the hundreds of software testing types. The different types of testing you can perform on a software is limited only by the degree of your imagination. Here, we provide you summary of some of the major ones.

LIST OF SOFTWARE TESTING TYPES
Type	Summary
Smoke Testing	Smoke Testing, also known as “Build Verification Testing”, is a type of software testing that comprises of a non-exhaustive set of tests that aim at ensuring that the most important functions work.
Functional Testing	Functional Testing is a type of software testing whereby the system is tested against the functional requirements/specifications.
Usability Testing	Usability Testing is a type of software testing done from an end-user’s perspective to determine if the system is easily usable.
Security Testing	Security Testing is a type of software testing that intends to uncover vulnerabilities of the system and determine that its data and resources are protected from possible intruders.
Performance Testing	Performance Testing is a type of software testing that intends to determine how a system performs in terms of responsiveness and stability under a certain load.
Regression Testing	Regression testing is a type of software testing that intends to ensure that changes (enhancements or defect fixes) to the software have not adversely affected it.
Compliance Testing	Compliance Testing [also known as conformance testing, regulation testing, standards testing] is a type of testing to determine the compliance of a system with internal or external standards.


