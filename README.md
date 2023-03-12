The Preliminary Result Database of Preventive Specific Bug Patterns
==== 



the preliminary results of applying phph to extract the bug pattern and the analysis of the results using tool phph.




<br><br>
About Defination of Prenventive Specific Bug Patterns and tools
------- 
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
Refenrence Paper：
[Ammonia: an approach for deriving project-specific bug patterns](https://link.springer.com/article/10.1007/s10664-020-09807-w)<br><br>

An Overview Figure to Show How Phph Works
Database formation of every phases OF Phph




About DataBase
------- 

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
About Repository
------- 
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



About Threshold to check bug Patterns
------- 
About Bugtype Classification
------- 
About ML-related Judgement
------- 
About Sample of Database
------- 
How to usen phph
------- 








