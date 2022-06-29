library(yaml)
library(ggplot2)
library(tidyverse)
library(rstan)
options(mc.cores = parallel::detectCores())

games = yaml::read_yaml("../games.yaml")
storytellers <- Reduce(union, lapply(games, function(game) game$storyteller))
goods <- Reduce(union, lapply(games, function(game) game$good))
evils <- Reduce(union, lapply(games, function(game) game$evil))
outcome <- sapply(games, function(game) game$winner == "good")
players <- union(goods, evils)

setdiff(players, evils)

length(players)
length(games)

mean(outcome)

good <- matrix(0, 15, length(games))
evil <- matrix(0, 5, length(games))

for (j in 1:length(games)) {
    game = games[[j]]
    for (i in 1:length(game$good)) {
        player = game$good[i]
        good[i, j] = match(player, players)
    }
    for (i in 1:length(game$evil)) {
        player = game$evil[i]
        evil[i, j] = match(player, players)
    }
}

data <- list(Nplayers = length(players),
             Ngames = length(games),
             good = good,
             evil = evil,
             outcome = outcome)

fit <- stan("model.stan", data = data)

1/mean(extract(fit)$sigma)

iq = (extract(fit)$skill * 15) + 100

skill <- as.data.frame(iq)
colnames(skill) <- players
skill$sample <- seq.int(nrow(skill))
skill <- pivot_longer(skill, !sample)

ggplot(skill) + aes(reorder(name, value), value) +
  geom_violin(linetype = "blank", fill = "#555555") + coord_flip() + theme_bw()

skill %>% group_by(name) %>%
    # summarise(mean = mean(value),
    #           lower = quantile(value, 0.05),
    #           upper = quantile(value, 0.95)) %>%
    summarise(mean = mean(value)) %>%
    arrange(-mean)

skill %>% group_by(sample) %>% summarise(max = max(value)) %>%
    right_join(skill) %>% group_by(name) %>% summarise(p = mean(value == max)) %>%
    arrange(-p)

# Example from here crashes:
# https://mc-stan.org/rstan/reference/stanmodel-method-optimizing.html
