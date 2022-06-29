data {
    int Nplayers;
    int Ngames;
    int good[15, Ngames];
    int evil[5, Ngames];
    int outcome[Ngames];
}

parameters {
    real skill[Nplayers];
    real<lower=0> sigma;
}

model {
    skill ~ normal(0, 1);
    sigma ~ gamma(1, 1);
    for (j in 1:Ngames) {
        int ngood = 0;
        real goodskill = 0;
        int nevil = 0;
        real evilskill = 0;

        for (i in 1:15) {
            if (good[i, j] == 0) break;
            goodskill += skill[good[i, j]];
            ngood += 1;
        }
        goodskill /= ngood;

        for (i in 1:5) {
            if (evil[i, j] == 0) break;
            evilskill += skill[evil[i, j]];
            nevil += 1;
        }
        evilskill /= nevil;

        outcome[j] ~ bernoulli_logit((goodskill - evilskill) * sigma);
    }
}
