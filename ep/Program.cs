using System;

namespace ep
{
    using System.IO;
    using System.Linq;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;
    using YamlDotNet.Serialization;
    using YamlDotNet.Serialization.NamingConventions;

    public class Game
    {
        public string Storyteller { get; set; }
        public string Edition { get; set; }
        public string[] Good { get; set; }
        public string[] Evil { get; set; }
        public string Winner { get; set; }
    }

    class Program
    {
        static (string[], string[]) LoadPlayers(Game[] games) {
            var players = new Dictionary<string,int>();
            var storytellers = new HashSet<string>();

            foreach (var game in games) {
                storytellers.Add(game.Storyteller);
                void addplayer(string player) {
                    if (!players.ContainsKey(player)) {
                        players[player] = 0;
                    }
                    players[player] += 1;
                }
                foreach (var player in game.Good) {
                    addplayer(player);
                }
                foreach (var player in game.Evil) {
                    addplayer(player);
                }
            }
            var players2 = players.Keys.Where(p => players[p] >= 1).Prepend("other").ToArray();
            return (storytellers.ToArray(), players2);
        }

        static void Main(string[] args)
        {
            // Model

            // Unkown players could be:
            // a. Fixed to skill 0
            // b. Treated as a random N(0, 1) sample (independent per player and game)
            // c. Treated as a single player with some inferred skill level
            // We use c, but can switch to a with the ConstrainEqual line below.
            // b is tricky to write down.

            // Open model evidence block
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);

            var Nplayers = Variable.New<int>();
            var player = new Range(Nplayers);
            var Ngames = Variable.New<int>();
            var game = new Range(Ngames);

            var bias = Variable.GaussianFromMeanAndVariance(0, 1);
            var skills = Variable.Array<double>(player).Named("skills");
            skills[player] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(player);
            // var scale = Variable.GammaFromShapeAndRate(1, 1);
            // skills[player] = Variable.GaussianFromMeanAndVariance(0, scale).ForEach(player);
            // Variable.ConstrainEqual(skills[0], 0);

            var outcome = Variable.Array<bool>(game).Named("outcome");
            var prediction = Variable.Array<bool>(game).Named("prediction");

            var goodTeamSize = Variable.Array<int>(game);
            var goodPlayer = new Range(goodTeamSize[game]);
            var goodTeam = Variable.Array(Variable.Array<int>(goodPlayer), game);

            var evilTeamSize = Variable.Array<int>(game);
            var evilPlayer = new Range(evilTeamSize[game]);
            var evilTeam = Variable.Array(Variable.Array<int>(evilPlayer), game);

            var factor = Variable.Array<double>(game);

            using (Variable.ForEach(game)) {
                var good = Variable.Sum(Variable.Subarray(skills, goodTeam[game])) / Variable.Double(goodTeamSize[game]);
                var evil = Variable.Sum(Variable.Subarray(skills, evilTeam[game])) / Variable.Double(evilTeamSize[game]);
                var diff = good - evil;
                // Deterministic
                // prediction[game] = diff > 0;
                // outcome[game] = diff > 0;
                // Logit
                // prediction[game] = Variable.Bernoulli(Variable.Logistic((diff + bias) * factor[game]));
                // outcome[game] = Variable.Bernoulli(Variable.Logistic((diff + bias) * factor[game]));
                // Probit
                prediction[game] = Variable.GaussianFromMeanAndVariance((diff + bias) * factor[game], 3.23) > 0;
                outcome[game] = Variable.GaussianFromMeanAndVariance((diff + bias) * factor[game], 3.23) > 0;
            }

            var iqs = Variable.Array<double>(player);
            using (Variable.ForEach(player)) {
                iqs[player] = skills[player]*15 + 100;
            }

            block.CloseBlock();

            // Load data
            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(HyphenatedNamingConvention.Instance)
                .Build();

            var yaml = File.ReadAllText("../games.yaml");
            var games = deserializer.Deserialize<Game[]>(yaml);
            var (storytellers, players) = LoadPlayers(games);

            Nplayers.ObservedValue = players.Length;
            Ngames.ObservedValue = games.Length;
            outcome.ObservedValue = games.Select(g => g.Winner == "good").ToArray();

            int playerIndex(string player) {
                return Enumerable.Contains(players, player) ? Array.IndexOf(players, player) : 0;
            }

            goodTeamSize.ObservedValue = games.Select(g => g.Good.Length).ToArray();
            goodTeam.ObservedValue = games.Select(g => g.Good.Select(playerIndex).ToArray()).ToArray();

            evilTeamSize.ObservedValue = games.Select(g => g.Evil.Length).ToArray();
            evilTeam.ObservedValue = games.Select(g => g.Evil.Select(playerIndex).ToArray()).ToArray();

            // Adjust the variance, as if we had calculated a per-player performance.
            factor.ObservedValue = games.Select(g => Math.Sqrt(1.0/g.Good.Length + 1.0/g.Evil.Length)).ToArray();

            // Run inference

            var engine = new InferenceEngine();
            // engine.NumberOfIterations = 1000;
            var ev = engine.Infer<Bernoulli>(evidence);
            Console.WriteLine("Evidence: {0}", Math.Exp(ev.LogOdds - Math.Log(0.5)*games.Length));
            var likelihoods = engine.Infer<Bernoulli[]>(prediction).Zip(outcome.ObservedValue)
                                    .Select(x => Math.Log(x.Second ? x.First.GetProbTrue() : x.First.GetProbFalse())).ToArray();
            Console.WriteLine("Likelihood: {0}", Math.Exp(likelihoods.Average()));
            Console.WriteLine("Accuracy: {0}", likelihoods.Select(p => Math.Exp(p) > 0.5).Count() / likelihoods.Length);
            // Console.WriteLine("Scale: {0}", Math.Sqrt(engine.Infer<Gamma>(scale).GetMean()));
            Console.WriteLine("Bias: {0}", engine.Infer<Gaussian>(bias));
            Console.WriteLine();
            foreach (var (name, skill) in players.Zip(engine.Infer<Gaussian[]>(iqs)).OrderBy(x => -x.Second.GetMean())) {
                Console.WriteLine("{0}: {1} ± {2}", name, Math.Round(skill.GetMean(), 1), Math.Round(Math.Sqrt(skill.GetVariance()), 1));
            }
        }
    }
}
