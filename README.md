# Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations

Daniel Brown*, Wonjoon Goo*, Prabhat Nagarajan, and Scott Niekum. (* Equal Contribution)

<p align=center>
  <img src='assets/figure.png' width=600>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/1904.06387">View on ArXiv</a> |
  <a href="https://hiwonjoon.github.io/ICML2019-TREX">Project Website</a>
</p>


This repository contains a code used to conduct experiments reported in the paper.

If you find this repository is useful in your research, please cite the paper:
```
@InProceedings{pmlr-v97-brown19a,
  title = {Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations},
  author = {Brown, Daniel and Goo, Wonjoon and Nagarajan, Prabhat and Niekum, Scott},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  pages = {783--792},
  year = {2019},
  editor = {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = {97},
  series = {Proceedings of Machine Learning Research},
  address = {Long Beach, California, USA},
  month = {09--15 Jun},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v97/brown19a/brown19a.pdf},
  url = {http://proceedings.mlr.press/v97/brown19a.html},
  abstract = {A critical flaw of existing inverse reinforcement learning (IRL) methods is their inability to significantly outperform the demonstrator. This is because IRL typically seeks a reward function that makes the demonstrator appear near-optimal, rather than inferring the underlying intentions of the demonstrator that may have been poorly executed in practice. In this paper, we introduce a novel reward-learning-from-observation algorithm, Trajectory-ranked Reward EXtrapolation (T-REX), that extrapolates beyond a set of (approximately) ranked demonstrations in order to infer high-quality reward functions from a set of potentially poor demonstrations. When combined with deep reinforcement learning, T-REX outperforms state-of-the-art imitation learning and IRL methods on multiple Atari and MuJoCo benchmark tasks and achieves performance that is often more than twice the performance of the best demonstration. We also demonstrate that T-REX is robust to ranking noise and can accurately extrapolate intention by simply watching a learner noisily improve at a task over time.}
}
```
