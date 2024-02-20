# RE-SORT: Removing Spurious Correlation in Multilevel Interaction for CTR Prediction

This is the official  implementation of the paper 《RE-SORT: Removing Spurious Correlation in Multi-level Interaction for CTR Prediction》

![Overview Framework](./re-sort.png)

## Datasets

1. [Criteo](./datasets/criteo)
2. [Avazu](./datasets/avazu)
3. [Frappe](./datasets/frappe)
4. [MovieLens](./datasets/movielens)
5. [UGC](./datasets/ugc)

## Training

python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

🔥 ## Citation

If you find our code or benchmarks helpful in your research, please cite the following paper:

```bibtex
@article{ Wu2024resort,
  Title = {RE-SORT: Removing Spurious Correlation in Multilevel Interaction for CTR Prediction},
  Author = {Songli, Wu and Liang, Du and Jia-Qi, Yang and Yuai, Wang and De-Chuan, Zhan and Shuang, Zhao and Zixun, Sun},
  Eprint = {arXiv preprint: 2309.14891},
  Year = {2024}
}
