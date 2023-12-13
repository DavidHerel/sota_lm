
# Ensemble of All

Repository containing [Ensemble of Language Models](https://arxiv.org/abs/2312.03735) on several LM benchmarks.

Thanks to this works one does not need to create an individual model that is the new state of the art to attract attention; it is sufficient to develop a new model that learns patterns which other models do not. Thus, even a suboptimal model can be found to have value.

## quick start

To run evaluation of code on all datasets:

```
$ python main.py
```

This will create optimal weighted combination of models in ensemble on each dataset. It will also generate `index.html` containing graphs and charts. 

[//]: # (It will also update `README.md` with the newest results.)

## results

Perplexity on validation and test with Ensemble of All:

| dataset      | valid ppl | test ppl |
|--------------|-----------|----------|
| penntreebank | 48.92     | 47.31    |
| wikitext-2   | 55.40     | 53.73    |
| wikitext-103 | 13.12     | 13.29    |

## addition of new models
New models can be easily add to the ensemble by producing probabilities of words on valid and test set and then they can be put into specific folders regarding datasets. We have reproduced most of the open sourced models and their probabilities on each word can be seen in individual folders `PennTreeBank`, `Wikitext-2`,`Wikitext-103`.

## citation

Please cite the [arxiv paper](https://arxiv.org/abs/2312.03735) if you use our work:

```bibtex
@misc{herel2023advancing,
      title={Advancing State of the Art in Language Modeling}, 
      author={David Herel and Tomas Mikolov},
      year={2023},
      eprint={2312.03735},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

## License
Our code is MIT licensed. See the **[LICENSE](https://github.com/DavidHerel/sota_lm/blob/main/LICENSE)** file for details.
