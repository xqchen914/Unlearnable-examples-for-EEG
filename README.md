# Unlearnable-examples-for-EEG
This repository is the implementation of privacy preserving user-wise perturbations from paper "User-wise Perturbations for User Identity Protection in EEG-Based BCIs". Please contact xqchen914@hust.edu.cn if you have any questions.

# Generate privcy unlearnable EEG data
```python
python main.py --model=EEGNet --dataset=MI2014001 --perturbation no rand sn optim_linf adv_linf
```
