# Bug Pattern Dataset
Dataset collecting by PHPH Tool 

# About Defination of Bug Patterns and tools




# About DataBase
5 tables for each database,basically for extracting and abstracting patterns

Chunks: Information of all change deltas

A code delta is a chunk of changed code. If a change is code addition, its chunk includes only after-change text. If a change is code deletion, its chunk includes only before-change text. If a change is code replacement, its chunk includes both before-change text and after-change text. 

Table chunks Elements
Pattern_hash:
A unique hash for each change pattern. Could consider it as a name of the way it change with same before-change text and same after-change text


之后补充每个TABLE元素含义
留白



Commits: Information of commits where at least an instance (an actual change) of the given change pattern appears because several instances of a change pattern can occur in the same commit.

Fragments: Texts after abstraction normalization but before transferred to Hash and Hashes listed one by one

Repositories: Information of project repository

Patterns: Group the chunks in same before-change-text(hash) and after-change text.

A change pattern is an abstract pattern that represents how source code was changed. A change pattern consists of code deltas whose both before-change text and after-change text are abstractly identical to one another. 
Reference:Ammonia: an approach for deriving project-specific bug patterns | SpringerLink

Table patterns
Type:
Change type
2 means a code deletion.
1 means a code addition.
0 means a code replacement.

SupportH:the number of instances included in a given change pattern. Do not know the meaning of H.
SupportC: the number of commits included in a given change pattern. 
Note that instances and commits are different because several instances of a change pattern can occur in the same commit.
The guess is proved after selecting more than 10 patterns and check the results
ConfidenceH: Confidence level in all the patterns in the same before-change text 
ConfidenceC: Confidence level

Confidence interval in all the patterns in the same before-change text
 (Proved by selection)
If the before text is null and the confidence looks incorrect ,there is no problem. Because in the paper
The proposed technique utilizes only code deletion and code replacement because code addition cannot be utilized to identify code fragments that include latent bugs.!

Code fragments in the target source code revision that match a particular change pattern in every chunk. Query could be the same in different matches.


# About Repository

Typer    53commits 922bugs GitHub - tiangolo/typer: Typer, build great CLIs. Easy to code. Based on Python type hints.

hummingbird 208commits 2733bugs ML-related | a library for compiling trained traditional ML models into tensor computations
GitHub - microsoft/hummingbird: Hummingbird compiles trained ML models into tensor computation for faster inference.
 
alibi-detect 196commits 3453bugs |a lightweight interactive visualization tool to help AI researchers discover correlations and patterns in high-dimensional data GitHub - SeldonIO/alibi-detect: Algorithms for outlier, adversarial and drift detection


mlem-By Iterative AI (creators of DVC). 172commits 5088bugs ML-relatedGitHub - iterative/mlem: 🐶 A tool to package, serve, and deploy any ML model on any platform.



# About Threshold to check bug Patterns

# About Bugtype Classification

# About ML-related Judgement

# About Sample of Database








