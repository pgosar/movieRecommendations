# pyright: basic

from fastapi import FastAPI

app = FastAPI()


@app.get("/api/python")
def recommend():
    import pandas as pd
    from fastai.collab import *
    from fastai.tabular.all import *

    path = untar_data(URLs.ML_100k)
    ratings = pd.read_csv(
        path / "u.data",
        delimiter="\t",
        header=None,
        names=["user", "movie", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        path / "u.item",
        delimiter="|",
        encoding="latin-1",
        usecols=(0, 1),
        names=("movie", "title"),
        header=None,
    )
    ratings = ratings.merge(movies)
    dls = CollabDataLoaders.from_df(ratings, item_name="title", bs=64)
    n_users = len(dls.classes["user"])
    n_movies = len(dls.classes["title"])
    n_factors = 5
    user_factors = torch.randn(n_users, n_factors)
    movie_factors = torch.randn(n_movies, n_factors)

    def create_params(size):
        return nn.Parameter(torch.zeros(*size).normal_(0, 0.1))

    class DotProductBias(Module):
        def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
            self.user_factors = create_params([n_users, n_factors])
            self.user_bias = create_params([n_users])
            self.movie_factors = create_params([n_movies, n_factors])
            self.movie_bias = create_params([n_movies])
            self.y_range = y_range

        def forward(self, x):
            users = self.user_factors[x[:, 0]]
            movies = self.movie_factors[x[:, 1]]
            res = (users * movies).sum(dim=1)
            res += self.user_bias[x[:, 0]] + self.movie_bias[x[:, 1]]
            return sigmoid_range(res, *self.y_range)

    model = DotProductBias(n_users, n_movies, 50)
    learn = Learner(dls, model, loss_func=MSELossFlat())
    learn.fit_one_cycle(5, 5e-3, wd=0.1)
    movie_bias = learn.model.movie_bias.squeeze()
    idxs = movie_bias.argsort()[:5]
    [dls.classes["title"][i] for i in idxs]
    idxs = movie_bias.argsort(descending=True)[:5]
    return [dls.classes["title"][i] for i in idxs]
