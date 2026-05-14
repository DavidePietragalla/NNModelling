# MNIST-FDS

This repo implements a simple CNN for classifying the MNIST digits.

It also shows how to use `wandb`, `hydra-conf`, and `lightning`.

## Technologies

In this project we use many technologies that you should try to be familiar with, as they'll make it easier for you to complete you projects.

### Be Tidy

It's crucial to keep you code organized and well-mantained, for this reason it's usually recommended to break your code into different files and folders, each containing only one class or a couple of functions, usually divided by functionality (try to avoid using generic names like `utils`).

For the same reason **avoid putting all your code into a jupyter notebook**, as this makes the code difficult to browse, and it quickly becomes unreasonably long and difficult to handle. Jupyter Notebooks are a nice tool for data analysis and exploration, and even for code running, but you should avoid keeping there all your code logic.

### Version Control

The most used and recommended tool for version control is [`git`](https://rogerdudler.github.io/git-guide/). Use it extensively to save your changes, share your code and edit together a project with your friends. It can be a bit difficult to get into it at first, but at the end you need to know only three things:

1. you `pull` the changes from the online repo.
2. you `commit` your local changes (possibly with meaningful descriptions).
3. you `push` the commits to the online repo.

When you want to collaborate with a friend on git remember that you may want to do so on different `branches` and then `merge` them together. If you don't want to delve into that, just remember to edit different files. Also Jupyter Notebooks don't play very nicely with git, so consider to clear all outputs out of the jupyter notebook before commiting the changes.

You can also enable the `GitLens` extension from VSCode, which hopefully should make it easier for you to handle git.

### Experiment Running

During a project it's likely that you'll run many experiments, each with a different set of parameters, ex: How deep is your network, or which non-linearity you want to use. You want to setup you code in such a way that you can easily run every model configuration with just a change of a couple of lines. Possibly try to keep all you configurable parameters in just one file, in such a way that if (for example) you want to change the value of the learning rate you don't need to open 5 different files to find its definition.

For this task there are many tools that you can use, and you may even build your own for small projects, but one tool is [`hydra-conf`](https://hydra.cc/) which enables you to write configurations in `yaml` files, compose them, and easily run multiple experiments at once. You can find an example of how to tun hydra experiments in `src/experiment.py` and configuration files in the `cfg` folder.

### Logging

For each of the experiments and relative parameters you want to keep a complete trace of whether it is completed, how it performed, which parameters you used. For this reasoning we highly encourage to set up some kind of tracing system, that allows you to record the outcome of every experiment.
This is imporant for multiple reasons:

- Keep track of previous experiments to avoid repeating them.
- Keep track of hyperparameters to analyse which one works best.
- Keep track of results in order to display them at the end.

One such tool is [**Weights & Biases**](https://wandb.ai/), a dashboard website in which you can upload the results of your experiments to confront them. It's also very nice for collaborative experiments, as everyone of your friends can easily upload their results without overwriting yours. Plus at the end you can just download the whole database of experiments in `csv` format to produce any kind of visualization you may like.

### Dependencies

```bash
uv sync
```

