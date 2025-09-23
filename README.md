# Reinforcement Learning for Secure Code Generation
This project aims to investigate the effectiveness of applying reinforcement learning to secure code generation. Specifically, we use an offline actor-critic approach, where the [actor](https://huggingface.co/Salesforce/codet5-large) model learns to generate secure code through the guidance of a second [critic](https://huggingface.co/Salesforce/codet5-base) model. The critic detects vulnerabilities in actor-generated code and provides dense feedback, which in turn guides the actor towards generating more secure code.

We utilize the [Devign](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) dataset to fine-tune the critic for vulnerability detection and the [ReposVul](https://github.com/Eshe0922/ReposVul) dataset to align the actor model with C++ code generation.  

## Overview 
![Overview of the Actor-Critic Reinforcement Learning Pipeline](images/RLPipeline.svg)

## Results

### Critic Fine-Tuning
| Model | Accuracy | F1 |
|-------|-------|-------|
| CodeT5-base | 63.61 | 60.49 |

### Actor Fine-Tuning
| Model | CodeBleu |
|-------|-------|
| CodeT5-large | 95.22 | 

### Actor-Critic Approach
| Metric | Result |
|-------|-------|
| VRR | 27.27 |
+ VRR = Vulnerability Reduction Rate, the relative amount of vulnerabilities repaired according to the critic



