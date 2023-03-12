# The Preliminary Result Database of Preventive Specific Bug Patterns

the preliminary results of applying phph to extract the bug pattern and the analysis of the results using tool phph.
<br><br>

Table of Contents
<br><br>
    * About Defination of Prenventive Specific Bug Patterns and tools<br>
    * About DataBase <br>
    * About Repository<br>
    * About Threshold to Check Bug Patterns<br>
    * About Bugtype Classification<br>
    * About ML-related Judgement<br>
    * About Sample of Database<br>
    * About How to use Phph Browse Server to Check the Details<br>
  
    


<br><br>
## About Defination of Prenventive Specific Bug Patterns and tools

There are several objectives to software maintenance. One of those objectives is to make sure that bugs in software are fixed. During the software development, finding and fixing buggy code is always an important and cost-intensive maintenance task.<br><br>
∎ Nowadays many kinds of preventive and corrective measures exist to help software practitioners use to perform it. Static analysis (SA) is one of the preventive techniques developers use, which warn developers about potential bugs by scanning their source code for commonly occurring bug patterns before they release the software.<br><br>
∎ Typically, SA tools scan for general bug patterns that are common to any software project (such as null pointer dereference), and not for project specific patterns.
However, past research has pointed to this lack of customizability as a severe limiting issue in SA.<br><br>
∎ A new approach called Ammonia, which is based on the methodology of SA identify changes across the development history of a project, as a means to identify project-specific bug patterns. They reflect the project as a whole and compliment the warnings from other SA tools that identify general bug patterns.<br><br>
∎ A new approach called Ammonia, which is based on the methodology of SA identify changes across the development history of a project, as a means to identify project-specific bug patterns. They reflect the project as a whole and compliment the warnings from other SA tools that identify general bug patterns.<br><br>
∎PHPH is an upgraded version of Ammonia, and compared to the previous one which mainly focused on extracting projects using JAVA language, it adds the ability to extract projects using python language, so that we can use it to analyze a large number of medium-sized projects in python language and machine learning projects.<br>
∎ We use phph.And this is the preliminary results of applying phph to extract the bug pattern and the analysis of the results<br><br>
Link for the Tool:<br>
[phph]( https://github.com/salab/phph/)<br>
[phph for python]( https://github.com/salab/phph/tree/merge-durun-phph)<br>
<br>
Reference Paper：
[Ammonia: an approach for deriving project-specific bug patterns](https://link.springer.com/article/10.1007/s10664-020-09807-w)<br><br>
<br>
An Overview Figure to Show How Phph Works<br><br>
![](https://github.com/IngridJiang/Bug-Pattern-Dataset/blob/main/An%20Overview%20Figure%20to%20Show%20How%20Phph%20Works.jpg) 
<br><br>Database formation of every phases OF Phph<br>
![](https://github.com/IngridJiang/Bug-Pattern-Dataset/blob/main/Database%20Formation%20of%20Every%20Phases%20of%20Phph.jpg) 


## About DataBase 

There are 5 tables for each database,basically for extracting and abstracting patterns.
<br> <br> 
1.Chunks: Information of all change deltas<br> 
A code delta is a chunk of changed code. If a change is code addition, its chunk includes only after-change text. If a change is code deletion, its chunk includes only before-change text. If a change is code replacement, its chunk includes both before-change text and after-change text. 
<br> <br> 
| Table Column | meaning |
| ------ | ------ |
| rowid | the id number of each row |
| id | the id number of each chunk |
| commit_id | the id number of each commit |
| file | the path of the source file where this code change occurs |
| old_begin | the row number of the beginning text of each chunk before changing |
| old_end | the row number of the end text of each chunk before changing |
| new_begin | the row number of the beginning text of each chunk after changing |
| new_end | the row number of the end text of each chunk after changing |
| pattern_hash | A unique hash for each change pattern. Could consider it as a name of the way it change with same before-change text and same after-change text |

<br> 

2.Commits: Information of commits where at least an instance (an actual change) of the given change pattern appears because several instances of a change pattern can occur in the same commit.
<br> <br> 
| Table Column | meaning |
| ------ | ------ |
| rowid | the id number of each row |
| id | the id number of each commit |
| hash | A unique hash for each commit. Could consider it as a name for different message |
| message | the path of the source file where this code change occurs |
| ignore | A boolean value.1 means it is not a bug-fix commit, while 0 means it is a bug-fix commit|

<br> 

3.Fragments: Texts after abstraction normalization but before transferred to Hash and Hashes listed one by one
<br> <br> 
| Table Column | meaning |
| ------ | ------ |
| rowid | the id number of each row |
| text | the after-abstraction text of each code fragment |
| hash | A unique hash for each fragment. Could consider it as a name for different texts |

<br>

4.Repositories: Information of project repository
<br> <br> 
| Table Column | meaning |
| ------ | ------ |
| rowid | the id number of each row |
| id | the id number of repository |
| url | URL for this repository |

<br>

5.Patterns: Group the chunks in same before-change-text(hash) and after-change text. <br> <br> 
A change pattern is an abstract pattern that represents how source code was changed. A change pattern consists of code deltas whose both before-change text and after-change text are abstractly identical to one another.  <br> <br> 
Reference:Ammonia: an approach for deriving project-specific bug patterns<br> 
https://link.springer.com/article/10.1007/s10664-020-09807-w<br> <br> 
| Table Column | meaning |
| ------ | ------ |
| rowid | the id number of each row |
| old | the hash of old text fragment |
| new | the hash of new text fragment |
| type | 2 means a code deletion.1 means a code addition.0 means a code replacement. |
| hash | A unique hash for each change pattern. Could consider it as a name of the way it change with same before-change text and same after-change text |
| supportH | the number of instances included in a given change pattern. |
| supportC | the number of commits included in a given change pattern |
| confidenceH | Confidence level in all the patterns in the same before-change text |
| confidenceC | Confidence level in all the patterns in the same before-change text |
| essential | Remarks |
| ignore | A boolean value.1 means it is not a bug-fix pattern, while 0 means it is a bug-fix pattern |
| bugtype | bugtype number of each pattern。For more information, see About Bugtype Classification below |
| mlrelated | A boolean value.1 means it is not ml-related, while 0 means it is ml-related |

<br> 
<br> 

Note:<br> 
1.Instances and commits are different because several instances of a change pattern can occur in the same commit.<br> 
2.Confidence interval in all the patterns in the same before-change text.We need this value because one of the Condition which preventive specific Bug Patterns should satisfy is that different patterns should have different before-text. <br> 


<br><br>
## About Repository

We choose four representative medium-sized repository databases for bug pattern extraction and classification as first try.More databases are available in the future。The details and links of the four databases are as follows：
<br><br>
1.[Typer](https://github.com/tiangolo/typer) 53commits 922bugs <br>
Typer is a library for building CLI applications that users will love using and developers will love creating. Based on Python 3.6+ type hints.  <br>   
2.[Hummingbird](https://github.com/microsoft/hummingbird)208commits 2733bugs<br>
Hummingbird is a library for compiling trained traditional ML models into tensor computations. Hummingbird allows users to seamlessly leverage neural network frameworks (such as PyTorch) to accelerate traditional ML models.      <br><br> 
3.[Alibi-detect](https://github.com/SeldonIO/alibi-detect)196commits 3453bugs<br>
Alibi Detect is an open source Python library focused on outlier, adversarial and drift detection. The package aims to cover both online and offline detectors for tabular data, text, images and time series. Both TensorFlow and PyTorch backends are supported for drift detection.       <br><br> 
4.[Mlem](https://github.com/iterative/mlem)172commits 5088bugs     <br>
By Iterative AI (creators of DVC). A tool to package, serve, and deploy any ML model on any platform.MLEM helps you package and deploy machine learning models. It saves ML models in a standard format that can be used in a variety of production scenarios such as real-time REST serving or batch processing.<br> <br> 



## About Threshold to Check Bug Patterns
According to the paper:Ammonia: an approach for deriving project-specific bug patterns,Change patterns that satisfy both the conditions remain.<br> <br>
Condition-1: change patterns related to bug-fix commits. Commits in the repository of the target software projects can be classified into bug-fix commits and other commits such as functional enhancement or refactoring. <br> <br>
Condition-2: change patterns whose before-texts are different from the before-texts of any other change patterns. We use only change patterns consisting of at least two changes and whose before-texts are different from the before-texts of all other change patterns.<br> <br>
The remaining change patterns ((a) that are part of a bug-fix commit, and (b) have identical after-change texts for all the changes) are used to identify latent problematic (buggy) code. We identify such change patterns as PSBPs.So in plain language, to get the initial data, we set the following thresholds.<br><br>
Condition-1: manually find bug-fix commits and only extract PSBPs from these commits. <br><br>
Condition-2: Concept: Confidence=1
             Real World not too strict threshold:Confidence>=0.90
<br><br>
In the process of extracting the bug pattern, it is recommended to use database tools such as SQLite p with it to reduce the workload and time spending, and the following is a simplified extraction process including some SQL statements.<br>
<br>

Phase 1:Satisfy Condition-2<br>
``` SQL
Update patterns Set ignore = 1 Where confidenceH< 0.90
``` 
<br>

Phase 2:Check commits to Satisfy Condition-1<br>
``` SQL
ALTER TABLE commits ADD ignore int DEFAULT 1
SELECT * FROM commits WHERE message like "%bug%" or "%fix%" or "%bug-fix%" ;
Update commits Set ignore=0 Where message like "%bug%" or "%fix%" or "%bug-fix%" or "%Fix%" or "%Bug-fix%" or "%Bug%";
``` 
Then manually check the message of bug-fix commits
<br><br>
Phase 3:Extract Patterns both Satisfy Condition-1 and Condition-2
``` SQL
Update patterns Set ignore = 2 where ignore=0 and hash in (select pattern_hash from chunks where commit_id in (select id from commits where ignore=0));
Update patterns Set ignore = 1 where ignore<>2;
Update patterns Set ignore = 0 where ignore=2;
``` 
<br>

Phase 4  manually check the patterns remained
``` SQL
SELECT * FROM patterns where ignore=0
``` 
<br>

Phase 5 Check the result
``` SQL
SELECT *,COUNT(*) FROM patterns where ignore=0
SELECT *,COUNT(*) FROM commits where ignore=0
```
<br>



 
## About Bugtype Classification
According to the paper:
[Not all bugs are the same: Understanding, characterizing, and classifying bug types](https://www.sciencedirect.com/science/article/abs/pii/S0164121219300536)
Bugtype are judged by 3 phases: <br>

### Judge by the bug report (Commit Message):9 types in total<br>
A.Configuration issue. <br>
The first category regards bugs concerned with building configuration files. Most of them are related to problems caused by 
(i) external libraries that should be updated or fixed and (ii) wrong directory or file paths in xml or manifest artifacts. <br>
Example summary. “JEE5 Web model does not update on changes in web.xml” 
Reason: Because it is mainly related to a wrong usage of external dependencies that cause issues in the web model of the application.  
<br>Bugtype Number:1
 <br><br>
B. Network issue. <br>
This category is related to bugs having connection or server issues, due to network problems, unexpected server shutdowns, or communication protocols that are not properly used within the source code. <br>
Example summary. “During a recent reorganization of code a couple of weeks ago, SSL recording no longer works” <br>
Reason: Due to a missing recording of the network traffic of the end-users of the project.
<br>Bugtype Number:2
<br><br>
C. Database-related issue. <br>
This category collects bugs that report problems with the connection between the main application and a database. <br>
Example summary. “Database connection stops action servlet from loading” <br>
Reason: Failed queries or connection, such as the case shown below where the developer reports a connection stop during the loading of a Java Servlet.
<br>Bugtype Number:3
<br><br>
D.GUI-related issue. <br>
This category refers to the possible bugs occurring within the Graphical User Interface (GUI) of a software project. It includes issues referring to (i) stylistic errors, i.e., screen layouts, elements colors and padding, text box appearance, and buttons, as well as (ii) unexpected failures appearing to the users in form of unusual error messages. <br>
Example summary. “Text when typing in input box is not viewable.”<br>
Reason: Because we do not see the actual text when we types in an input field.
<br>Bugtype Number:4
<br><br>
E. Performance issue. <br>
This category collects bugs that report performance issues, including memory overuse, energy leaks, and methods causing endless loops. <br>
Example summary. “Loading a large script in the Rhino debugger results in an endless loop (100% CPU utilization)”<br>
Reason: Due to the difficulties in loading an external file.
<br>Bugtype Number:5
<br><br>
F. Permission/deprecation issue. <br>
Bugs in this category are related to two main causes: on the one hand, they are due to the presence, modification, or removal of deprecated method calls or APIs; on the other hand, problems related to unused API permissions are included. <br>
Example summary. “setTrackModification(boolean) not deprecated; but does not work” <br>
“Access violation in DOMServices::getNamespaceForPrefix (DOMServices.cpp:759)”  <br>
Reason: The first involves a bug appearing in the case of an unexpected behavior when the method of an external API is called. The second mentions a bug that appears through malformed communication with an API. 
<br>Bugtype Number:6
<br><br>
G. Security issue. <br>
Vulnerability and other security-related problems are included in this category. These types of bugs usually refer to reload certain parameters and removal of unused permissions that might decrease the overall reliability of the system. <br>
Example summary. “Disable cocoon reload parameter for security reasons” <br>
Reason: The Cocoon framework was temporarily stopped because of a potential vulnerability.
<br>Bugtype Number:7
<br><br>
H. Program anomaly issue. <br>
Bugs introduced by developers when enhancing existing source code, and that are concerned with specific circumstances such as exceptions, problems with return values, and unexpected crashes due to issues in the logic (rather than, e.g., the GUI) of the program. <br>
Example summary. “Program terminates prematurely before all execution events are loaded in the model”
<br>Bugtype Number:8
<br><br>
I. Test code-related issue. <br>
The last category is concerned with bugs appearing in test code. Looking at bug reports in this category, we observed that they usually report problems due to (i) running, fixing, or updating test cases, (ii) intermittent tests, and (iii) the inability of a test to find de-localized bugs. <br>
Example summary. “[the test] makes mochitest-plain time out when the HTML5 parser is enabled” <br>
Reason: Because of a wrong usage of mocking. 
<br>Bugtype Number:9
<br><br>
### Topics analysis. 
Result got by the paper using LDA-GA algorithm over the bug reports of each category of bug type present in their taxonomy. 
LDA-GA found up to five different clusters that describe the topics characterizing each bug type; a ‘–’ symbol is put in the table in case LDA-GA did not identify more topics for a certain bug type.<br>
![](https://github.com/IngridJiang/Bug-Pattern-Dataset/blob/main/Bugtype%20Topic.png) 
<br>
### By looking at code changes
By looking at code changes itself,we could judge the bug type.
<br><br>





## About ML-related Judgement

ML-related value is got by investigating whether the instances of each BUG pattern are all related to machine learning. The criterion to determine whether it is related to machine learning is to check whether it imports machine learning llibraries. This requires a list of machine learning-related libraries.<br>
About the list of ML libraries,we refers to [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning).Thanks to all the contributers to this curated list of awesome Machine Learning frameworks, libraries and software and it is still keep updating.<br>
As an initial database, we only consider the part of Python-General-Purpose Machine Learning, and consider adding it if there are other needs.If all instances of each pattern import ML libraries, the mlrelated value of this pattern could be 1.<br>

### Python General-Purpose Machine Learning

 * [RexMex](https://github.com/AstraZeneca/rexmex) -> A general purpose recommender metrics library for fair evaluation.
 * [ChemicalX](https://github.com/AstraZeneca/chemicalx) -> A PyTorch based deep learning library for drug pair scoring
 * [Microsoft ML for Apache Spark](https://github.com/Azure/mmlspark) -> A distributed machine learning framework Apache Spark
 * [Shapley](https://github.com/benedekrozemberczki/shapley) -> A data-driven framework to quantify the value of classifiers in a machine learning ensemble.
 * [igel](https://github.com/nidhaloff/igel) -> A delightful machine learning tool that allows you to train/fit, test and use models **without writing code**
 * [ML Model building](https://github.com/Shanky-21/Machine_learning) -> A Repository Containing Classification, Clustering, Regression, Recommender Notebooks with illustration to make them.
 * [ML/DL project template](https://github.com/PyTorchLightning/deep-learning-project-template)
 * [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) -> A temporal extension of PyTorch Geometric for dynamic graph representation learning.
 * [Little Ball of Fur](https://github.com/benedekrozemberczki/littleballoffur) -> A graph sampling extension library for NetworkX with a Scikit-Learn like API.
 * [Karate Club](https://github.com/benedekrozemberczki/karateclub) -> An unsupervised machine learning extension library for NetworkX with a Scikit-Learn like API.
* [Auto_ViML](https://github.com/AutoViML/Auto_ViML) -> Automatically Build Variant Interpretable ML models fast! Auto_ViML is pronounced "auto vimal", is a comprehensive and scalable Python AutoML toolkit with imbalanced handling, ensembling, stacking and built-in feature selection. Featured in <a href="https://towardsdatascience.com/why-automl-is-an-essential-new-tool-for-data-scientists-2d9ab4e25e46?source=friends_link&sk=d03a0cc55c23deb497d546d6b9be0653">Medium article</a>.
* [PyOD](https://github.com/yzhao062/pyod) -> Python Outlier Detection, comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. Featured for Advanced models, including Neural Networks/Deep Learning and Outlier Ensembles.
* [steppy](https://github.com/neptune-ml/steppy) -> Lightweight, Python library for fast and reproducible machine learning experimentation. Introduces a very simple interface that enables clean machine learning pipeline design.
* [steppy-toolkit](https://github.com/neptune-ml/steppy-toolkit) -> Curated collection of the neural networks, transformers and models that make your machine learning work faster and more effective.
* [CNTK](https://github.com/Microsoft/CNTK) - Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit. Documentation can be found [here](https://docs.microsoft.com/cognitive-toolkit/).
* [Couler](https://github.com/couler-proj/couler) - Unified interface for constructing and managing machine learning workflows on different workflow engines, such as Argo Workflows, Tekton Pipelines, and Apache Airflow.
* [auto_ml](https://github.com/ClimbsRocks/auto_ml) - Automated machine learning for production and analytics. Lets you focus on the fun parts of ML, while outputting production-ready code, and detailed analytics of your dataset and results. Includes support for NLP, XGBoost, CatBoost, LightGBM, and soon, deep learning.
* [dtaidistance](https://github.com/wannesm/dtaidistance) - High performance library for time series distances (DTW) and time series clustering.
* [einops](https://github.com/arogozhnikov/einops) - Deep learning operations reinvented (for pytorch, tensorflow, jax and others).
* [machine learning](https://github.com/jeff1evesque/machine-learning) - automated build consisting of a [web-interface](https://github.com/jeff1evesque/machine-learning#web-interface), and set of [programmatic-interface](https://github.com/jeff1evesque/machine-learning#programmatic-interface) API, for support vector machines. Corresponding dataset(s) are stored into a SQL database, then generated model(s) used for prediction(s), are stored into a NoSQL datastore.
* [XGBoost](https://github.com/dmlc/xgboost) - Python bindings for eXtreme Gradient Boosting (Tree) Library.
* [ChefBoost](https://github.com/serengil/chefboost) - a lightweight decision tree framework for Python with categorical feature support covering regular decision tree algorithms such as ID3, C4.5, CART, CHAID and regression tree; also some advanved bagging and boosting techniques such as gradient boosting, random forest and adaboost.
* [Apache SINGA](https://singa.apache.org) - An Apache Incubating project for developing an open source machine learning library.
* [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - Book/iPython notebooks on Probabilistic Programming in Python.
* [Featureforge](https://github.com/machinalis/featureforge) A set of tools for creating and testing machine learning features, with a scikit-learn compatible API.
* [MLlib in Apache Spark](http://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) - a service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [Towhee](https://towhee.io) - A Python module that encode unstructured data into embeddings.
* [scikit-learn](https://scikit-learn.org/) - A Python module for machine learning built on top of SciPy.
* [metric-learn](https://github.com/metric-learn/metric-learn) - A Python module for metric learning.
* [OpenMetricLearning](https://github.com/OML-Team/open-metric-learning) - A PyTorch-based framework to train and validate the models producing high-quality embeddings.
* [Intel(R) Extension for Scikit-learn](https://github.com/intel/scikit-learn-intelex) - A seamless way to speed up your Scikit-learn applications with no accuracy loss and code changes.
* [SimpleAI](https://github.com/simpleai-team/simpleai) Python implementation of many of the artificial intelligence algorithms described in the book "Artificial Intelligence, a Modern Approach". It focuses on providing an easy to use, well documented and tested library.
* [astroML](https://www.astroml.org/) - Machine Learning and Data Mining for Astronomy.
* [graphlab-create](https://turi.com/products/create/docs/) - A library with various machine learning models (regression, clustering, recommender systems, graph analytics, etc.) implemented on top of a disk-backed DataFrame.
* [BigML](https://bigml.com) - A library that contacts external servers.
* [pattern](https://github.com/clips/pattern) - Web mining module for Python.
* [NuPIC](https://github.com/numenta/nupic) - Numenta Platform for Intelligent Computing.
* [Pylearn2](https://github.com/lisa-lab/pylearn2) - A Machine Learning library based on [Theano](https://github.com/Theano/Theano). **[Deprecated]**
* [keras](https://github.com/keras-team/keras) - High-level neural networks frontend for [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/CNTK) and [Theano](https://github.com/Theano/Theano).
* [Lasagne](https://github.com/Lasagne/Lasagne) - Lightweight library to build and train neural networks in Theano.
* [hebel](https://github.com/hannes-brt/hebel) - GPU-Accelerated Deep Learning Library in Python. **[Deprecated]**
* [Chainer](https://github.com/chainer/chainer) - Flexible neural network framework.
* [prophet](https://facebook.github.io/prophet/) - Fast and automated time series forecasting framework by Facebook.
* [gensim](https://github.com/RaRe-Technologies/gensim) - Topic Modelling for Humans.
* [topik](https://github.com/ContinuumIO/topik) - Topic modelling toolkit. **[Deprecated]**
* [PyBrain](https://github.com/pybrain/pybrain) - Another Python Machine Learning Library.
* [Brainstorm](https://github.com/IDSIA/brainstorm) - Fast, flexible and fun neural networks. This is the successor of PyBrain.
* [Surprise](https://surpriselib.com) - A scikit for building and analyzing recommender systems.
* [implicit](https://implicit.readthedocs.io/en/latest/quickstart.html) - Fast Python Collaborative Filtering for Implicit Datasets.
* [LightFM](https://making.lyst.com/lightfm/docs/home.html) -  A Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.
* [Crab](https://github.com/muricoca/crab) - A flexible, fast recommender engine. **[Deprecated]**
* [python-recsys](https://github.com/ocelma/python-recsys) - A Python library for implementing a Recommender System.
* [thinking bayes](https://github.com/AllenDowney/ThinkBayes) - Book on Bayesian Analysis.
* [Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/williamFalcon/pix2pix-keras) - Implementation of image to image (pix2pix) translation from the paper by [isola et al](https://arxiv.org/pdf/1611.07004.pdf).[DEEP LEARNING]
* [Restricted Boltzmann Machines](https://github.com/echen/restricted-boltzmann-machines) -Restricted Boltzmann Machines in Python. [DEEP LEARNING]
* [Bolt](https://github.com/pprett/bolt) - Bolt Online Learning Toolbox. **[Deprecated]**
* [CoverTree](https://github.com/patvarilly/CoverTree) - Python implementation of cover trees, near-drop-in replacement for scipy.spatial.kdtree **[Deprecated]**
* [nilearn](https://github.com/nilearn/nilearn) - Machine learning for NeuroImaging in Python.
* [neuropredict](https://github.com/raamana/neuropredict) - Aimed at novice machine learners and non-expert programmers, this package offers easy (no coding needed) and comprehensive machine learning (evaluation and full report of predictive performance WITHOUT requiring you to code) in Python for NeuroImaging and any other type of features. This is aimed at absorbing much of the ML workflow, unlike other packages like nilearn and pymvpa, which require you to learn their API and code to produce anything useful.
* [imbalanced-learn](https://imbalanced-learn.org/stable/) - Python module to perform under sampling and oversampling with various techniques.
* [imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble) - Python toolbox for quick implementation, modification, evaluation, and visualization of ensemble learning algorithms for class-imbalanced data. Supports out-of-the-box multi-class imbalanced (long-tailed) classification.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox.
* [Pyevolve](https://github.com/perone/Pyevolve) - Genetic algorithm framework. **[Deprecated]**
* [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [breze](https://github.com/breze-no-salt/breze) - Theano based library for deep and recurrent neural networks.
* [Cortex](https://github.com/cortexlabs/cortex) - Open source platform for deploying machine learning models in production.
* [pyhsmm](https://github.com/mattjj/pyhsmm) - library for approximate unsupervised inference in Bayesian Hidden Markov Models (HMMs) and explicit-duration Hidden semi-Markov Models (HSMMs), focusing on the Bayesian Nonparametric extensions, the HDP-HMM and HDP-HSMM, mostly with weak-limit approximations.
* [SKLL](https://github.com/EducationalTestingService/skll) - A wrapper around scikit-learn that makes it simpler to conduct experiments.
* [neurolab](https://github.com/zueve/neurolab)
* [Spearmint](https://github.com/HIPS/Spearmint) - Spearmint is a package to perform Bayesian optimization according to the algorithms outlined in the paper: Practical Bayesian Optimization of Machine Learning Algorithms. Jasper Snoek, Hugo Larochelle and Ryan P. Adams. Advances in Neural Information Processing Systems, 2012. **[Deprecated]**
* [Pebl](https://github.com/abhik/pebl/) - Python Environment for Bayesian Learning. **[Deprecated]**
* [Theano](https://github.com/Theano/Theano/) - Optimizing GPU-meta-programming code generating array oriented optimizing math compiler in Python.
* [TensorFlow](https://github.com/tensorflow/tensorflow/) - Open source software library for numerical computation using data flow graphs.
* [pomegranate](https://github.com/jmschrei/pomegranate) - Hidden Markov Models for Python, implemented in Cython for speed and efficiency.
* [python-timbl](https://github.com/proycon/python-timbl) - A Python extension module wrapping the full TiMBL C++ programming interface. Timbl is an elaborate k-Nearest Neighbours machine learning toolkit.
* [deap](https://github.com/deap/deap) - Evolutionary algorithm framework.
* [pydeep](https://github.com/andersbll/deeppy) - Deep Learning In Python. **[Deprecated]**
* [mlxtend](https://github.com/rasbt/mlxtend) - A library consisting of useful tools for data science and machine learning tasks.
* [neon](https://github.com/NervanaSystems/neon) - Nervana's [high-performance](https://github.com/soumith/convnet-benchmarks) Python-based Deep Learning framework [DEEP LEARNING]. **[Deprecated]**
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search.
* [Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning) - Code samples for my book "Neural Networks and Deep Learning" [DEEP LEARNING].
* [Annoy](https://github.com/spotify/annoy) - Approximate nearest neighbours implementation.
* [TPOT](https://github.com/EpistasisLab/tpot) - Tool that automatically creates and optimizes machine learning pipelines using genetic programming. Consider it your personal data science assistant, automating a tedious part of machine learning.
* [pgmpy](https://github.com/pgmpy/pgmpy) A python library for working with Probabilistic Graphical Models.
* [DIGITS](https://github.com/NVIDIA/DIGITS) - The Deep Learning GPU Training System (DIGITS) is a web application for training deep learning models.
* [Orange](https://orange.biolab.si/) - Open source data visualization and data analysis for novices and experts.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [milk](https://github.com/luispedro/milk) - Machine learning toolkit focused on supervised classification. **[Deprecated]**
* [TFLearn](https://github.com/tflearn/tflearn) - Deep learning library featuring a higher-level API for TensorFlow.
* [REP](https://github.com/yandex/rep) - an IPython-based environment for conducting data-driven research in a consistent and reproducible way. REP is not trying to substitute scikit-learn, but extends it and provides better user experience. **[Deprecated]**
* [rgf_python](https://github.com/RGF-team/rgf) - Python bindings for Regularized Greedy Forest (Tree) Library.
* [skbayes](https://github.com/AmazaspShumik/sklearn-bayes) - Python package for Bayesian Machine Learning with scikit-learn API.
* [fuku-ml](https://github.com/fukuball/fuku-ml) - Simple machine learning library, including Perceptron, Regression, Support Vector Machine, Decision Tree and more, it's easy to use and easy to learn for beginners.
* [Xcessiv](https://github.com/reiinakano/xcessiv) - A web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling.
* [PyTorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python with strong GPU acceleration
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - The lightweight PyTorch wrapper for high-performance AI research.
* [PyTorch Lightning Bolts](https://github.com/PyTorchLightning/pytorch-lightning-bolts) - Toolbox of models, callbacks, and datasets for AI/ML researchers.
* [skorch](https://github.com/skorch-dev/skorch) - A scikit-learn compatible neural network library that wraps PyTorch.
* [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) - Implementations of Machine Learning models from scratch in Python with a focus on transparency. Aims to showcase the nuts and bolts of ML in an accessible way.
* [Edward](http://edwardlib.org/) - A library for probabilistic modelling, inference, and criticism. Built on top of TensorFlow.
* [xRBM](https://github.com/omimo/xRBM) - A library for Restricted Boltzmann Machine (RBM) and its conditional variants in Tensorflow.
* [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, well documented and supports CPU and GPU (even multi-GPU) computation.
* [stacked_generalization](https://github.com/fukatani/stacked_generalization) - Implementation of machine learning stacking technique as a handy library in Python.
* [modAL](https://github.com/modAL-python/modAL) - A modular active learning framework for Python, built on top of scikit-learn.
* [Cogitare](https://github.com/cogitare-ai/cogitare): A Modern, Fast, and Modular Deep Learning and Machine Learning framework for Python.
* [Parris](https://github.com/jgreenemi/Parris) - Parris, the automated infrastructure setup tool for machine learning algorithms.
* [neonrvm](https://github.com/siavashserver/neonrvm) - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [Turi Create](https://github.com/apple/turicreate) - Machine learning from Apple. Turi Create simplifies the development of custom machine learning models. You don't have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.
* [xLearn](https://github.com/aksnzhy/xlearn) - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertisement and recommender systems.
* [mlens](https://github.com/flennerhag/mlens) - A high performance, memory efficient, maximally parallelized ensemble learning, integrated with scikit-learn.
* [Thampi](https://github.com/scoremedia/thampi) - Machine Learning Prediction System on AWS Lambda
* [MindsDB](https://github.com/mindsdb/mindsdb) - Open Source framework to streamline use of neural networks.
* [Microsoft Recommenders](https://github.com/Microsoft/Recommenders): Examples and best practices for building recommendation systems, provided as Jupyter notebooks. The repo contains some of the latest state of the art algorithms from Microsoft Research as well as from other companies and institutions.
* [StellarGraph](https://github.com/stellargraph/stellargraph): Machine Learning on Graphs, a Python library for machine learning on graph-structured (network-structured) data.
* [BentoML](https://github.com/bentoml/bentoml): Toolkit for package and deploy machine learning models for serving in production
* [MiraiML](https://github.com/arthurpaulino/miraiml): An asynchronous engine for continuous & autonomous machine learning, built for real-time usage.
* [numpy-ML](https://github.com/ddbourgin/numpy-ml): Reference implementations of ML models written in numpy
* [Neuraxle](https://github.com/Neuraxio/Neuraxle): A framework providing the right abstractions to ease research, development, and deployment of your ML pipelines.
* [Cornac](https://github.com/PreferredAI/cornac) - A comparative framework for multimodal recommender systems with a focus on models leveraging auxiliary data.
* [JAX](https://github.com/google/jax) - JAX is Autograd and XLA, brought together for high-performance machine learning research.
* [Catalyst](https://github.com/catalyst-team/catalyst) - High-level utils for PyTorch DL & RL research. It was developed with a focus on reproducibility, fast experimentation and code/ideas reusing. Being able to research/develop something new, rather than write another regular train loop.
* [Fastai](https://github.com/fastai/fastai) - High-level wrapper built on the top of Pytorch which supports vision, text, tabular data and collaborative filtering.
* [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow) - A machine learning framework for multi-output/multi-label and stream data.
* [Lightwood](https://github.com/mindsdb/lightwood) - A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with objective to build predictive models with one line of code.
* [bayeso](https://github.com/jungtaekkim/bayeso) - A simple, but essential Bayesian optimization package, written in Python.
* [mljar-supervised](https://github.com/mljar/mljar-supervised) - An Automated Machine Learning (AutoML) python package for tabular data. It can handle: Binary Classification, MultiClass Classification and Regression. It provides explanations and markdown reports.
* [evostra](https://github.com/alirezamika/evostra) - A fast Evolution Strategy implementation in Python.
* [Determined](https://github.com/determined-ai/determined) - Scalable deep learning training platform, including integrated support for distributed training, hyperparameter tuning, experiment tracking, and model management.
* [PySyft](https://github.com/OpenMined/PySyft) - A Python library for secure and private Deep Learning built on PyTorch and TensorFlow.
* [PyGrid](https://github.com/OpenMined/PyGrid/) - Peer-to-peer network of data owners and data scientists who can collectively train AI models using PySyft
* [sktime](https://github.com/alan-turing-institute/sktime) - A unified framework for machine learning with time series
* [OPFython](https://github.com/gugarosa/opfython) - A Python-inspired implementation of the Optimum-Path Forest classifier.
* [Opytimizer](https://github.com/gugarosa/opytimizer) - Python-based meta-heuristic optimization techniques.
* [Gradio](https://github.com/gradio-app/gradio) - A Python library for quickly creating and sharing demos of models. Debug models interactively in your browser, get feedback from collaborators, and generate public links without deploying anything.
* [Hub](https://github.com/activeloopai/Hub) - Fastest unstructured dataset management for TensorFlow/PyTorch. Stream & version-control data. Store even petabyte-scale data in a single numpy-like array on the cloud accessible on any machine. Visit [activeloop.ai](https://activeloop.ai) for more info.
* [Synthia](https://github.com/dmey/synthia) - Multidimensional synthetic data generation in Python.
* [ByteHub](https://github.com/bytehub-ai/bytehub) - An easy-to-use, Python-based feature store. Optimized for time-series data.
* [Backprop](https://github.com/backprop-ai/backprop) - Backprop makes it simple to use, finetune, and deploy state-of-the-art ML models.
* [River](https://github.com/online-ml/river): A framework for general purpose online machine learning.
* [FEDOT](https://github.com/nccr-itmo/FEDOT): An AutoML framework for the automated design of composite modelling pipelines. It can handle classification, regression, and time series forecasting tasks on different types of data (including multi-modal datasets).
* [Sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt): An AutoML package for hyperparameters tuning using evolutionary algorithms, with built-in callbacks, plotting, remote logging and more.
* [Evidently](https://github.com/evidentlyai/evidently): Interactive reports to analyze machine learning models during validation or production monitoring.
* [Streamlit](https://github.com/streamlit/streamlit): Streamlit is an framework to create beautiful data apps in hours, not weeks.
* [Optuna](https://github.com/optuna/optuna): Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning.
* [Deepchecks](https://github.com/deepchecks/deepchecks): Validation & testing of machine learning models and data during model development, deployment, and production. This includes checks and suites related to various types of issues, such as model performance, data integrity, distribution mismatches, and more.
* [Shapash](https://github.com/MAIF/shapash) : Shapash is a Python library that provides several types of visualization that display explicit labels that everyone can understand.
* [Eurybia](https://github.com/MAIF/eurybia): Eurybia monitors data and model drift over time and securizes model deployment with data validation.
* [Colossal-AI](https://github.com/hpcaitech/ColossalAI): An open-source deep learning system for large-scale model training and inference with high efficiency and low cost.
* [dirty_cat](https://github.com/dirty-cat/dirty_cat) - facilitates machine-learning on dirty, non-curated categories. It provides transformers and encoders robust to morphological variants, such as typos.
* [Upgini](https://github.com/upgini/upgini): Free automated data & feature enrichment library for machine learning - automatically searches through thousands of ready-to-use features from public and community shared data sources and enriches your training dataset with only the accuracy improving features.
* [AutoML-Implementation-for-Static-and-Dynamic-Data-Analytics](https://github.com/Western-OC2-Lab/AutoML-Implementation-for-Static-and-Dynamic-Data-Analytics): A tutorial to help machine learning researchers to automatically obtain optimized machine learning models with the optimal learning performance on any specific task.
* [SKBEL](https://github.com/robinthibaut/skbel): A Python library for Bayesian Evidential Learning (BEL) in order to estimate the uncertainty of a prediction.
* [NannyML](https://bit.ly/nannyml-github-machinelearning): Python library capable of fully capturing the impact of data drift on performance. Allows estimation of post-deployment model performance without access to targets.
* [cleanlab](https://github.com/cleanlab/cleanlab): The standard data-centric AI package for data quality and machine learning with messy, real-world data and labels.
* [AutoGluon](https://github.com/awslabs/autogluon): AutoML for Image, Text, Tabular, Time-Series, and MultiModal Data.


## About Sample of Database
Randomly filter 300 data for each database and check the bug-fix ones whose ignore values equal 0 by SQL statement and put the results in this repository.
```SQL
select * from patterns order by RANDOM()limit 300;
```
<br>

## About How to use Phph Browse Server to Check the Details
If you need to extract Java project, please refer to read me.md of [phph]( https://github.com/salab/phph/).<br>
To extract python projects:<br>
### Build
```
$ git clone https://github.com/salab/phph.git
$ cd phph
$ git branch-a|cat
$ git checkout merge-durun-phph
$ git submodule init
$ git submodule update
$ ./gradlew shadowJar
$ java -jar build/libs/phph-all.jar <cmd> [options...]
```
<br>

### Usage
```
$ java -jar phph-all.jar init                                    # initalize database
$ java -jar phph-all.jar extract --splitter=python3 --repository=/path/to/repo/.git # extract patterns
$ java -jar phph-all.jar find                                    # find pattern application opportinuties
$ java -jar phph-all.jar measure                                 # compute metric values
```
<br>

### Browse Server
```
$ java -jar phph-all.jar browse                                            
```
<br><br>













