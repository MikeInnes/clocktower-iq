# Clocktower IQ

This code implements a simple player ranking system for [Blood on the Clocktower](https://www.kickstarter.com/projects/pandemoniuminstitute/blood-on-the-clocktower), modelled on [TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/). For technical details see the [blog post](https://mikeinnes.io/2022/06/29/clocktower.html).

## Data format

Games are stored in `games.yaml` in the following (hopefully fairly self explanatory) format.

```yaml
- storyteller: Bee
  edition: trouble-brewing
  good:
    - Mole
    - Lizard
    - Mouse
    - Lemur
    - Kangaroo
  evil:
    - Giraffe
    - Monkey
  winner: evil
```

## Infer.NET code

The `ep` folder contains a fast inference algorithm using Infer.NET. You'll need to have the `dotnet` command installed (use `brew install --cask dotnet-sdk` on mac with homebrew). Run `dotnet restore` to grab dependencies, then `dotnet run` to run inference. You'll get output like this:

```
Compiling model...done.
Iterating:
.........|.........|.........|.........|.........| 50
Evidence: 1.6255148666201176
Likelihood: 0.5734421123663153
Accuracy: 1
Bias: Gaussian(-0.7407, 0.2539)

Bee: 109.1 ± 13.2
Lizard: 108 ± 13.3
Giraffe: 104.6 ± 14.7
Shrimp: 103.7 ± 14.7
...
```

The code outputs a few model fit metrics followed by a rating for each player. Each player's IQ is followed by an uncertainty (the standard deviation). To get a 95% confidence interval add/subtract twice the standard deviation, eg `109.1 ± 13.2` => `[82.7, 135.5]`. If you want to rank players, it's best to use the 95% confidence lower bound.

## R/Stan code

The `stan` folder replicates the results in Stan. This is mostly only useful if your lap is getting chilly, but if you step through the code it also generates a nice violin plot.

The R code uses `renv`, so you'll need to use `renv::restore()` to grab dependencies before running.
