import numpy as np, pandas as pd, warnings, importlib, subprocess, sys
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy.stats import poisson
import xgboost as xgb
import ipywidgets as widgets

matches_raw = pd.read_csv("matches_raw.csv")
matches = matches_raw[["HomeTeam","AwayTeam","HomeGoals","AwayGoals"]]

if importlib.util.find_spec("lightgbm") is None:
    print("Installing LightGBM … (one-time, 20-30 s)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])
import lightgbm as lgb

req = {"HomeTeam","AwayTeam","HomeGoals","AwayGoals"}
if not req.issubset(matches.columns):
    raise ValueError(f"`matches` missing {req - set(matches.columns)}")

###############################################################################
# 1.  Poisson GLM baseline                                                     #
###############################################################################
gld = pd.concat([
    matches[["HomeTeam","AwayTeam","HomeGoals"]]
        .assign(home=1)
        .rename(columns={"HomeTeam":"team","AwayTeam":"opponent","HomeGoals":"goals"}),
    matches[["AwayTeam","HomeTeam","AwayGoals"]]
        .assign(home=0)
        .rename(columns={"AwayTeam":"team","HomeTeam":"opponent","AwayGoals":"goals"})
])
pois_mdl = smf.glm("goals ~ home + team + opponent", data=gld,
                   family=sm.families.Poisson()).fit()

def predict_pois(home, away, max_g=8):
    mu_h = pois_mdl.predict({"team":[home],"opponent":[away],"home":[1]})[0]
    mu_a = pois_mdl.predict({"team":[away],"opponent":[home],"home":[0]})[0]
    return _prob_table(mu_h, mu_a, max_g)

###############################################################################
# 2.  XGBoost Poisson                                                          #
###############################################################################
X_oh = pd.get_dummies(matches[["HomeTeam","AwayTeam"]]); X_oh["home"]=1
dmat = xgb.DMatrix(X_oh)
params = dict(objective="count:poisson", eval_metric="poisson-nloglik",
              eta=.05, max_depth=5, subsample=.8, colsample_bytree=.8)
xgb_h = xgb.train(params, xgb.DMatrix(X_oh, matches.HomeGoals), 600, verbose_eval=False)
xgb_a = xgb.train(params, xgb.DMatrix(X_oh, matches.AwayGoals), 600, verbose_eval=False)

def predict_xgb(home, away, max_g=8):
    vec = _vectorize(home, away, X_oh.columns)
    mu_h = float(xgb_h.predict(xgb.DMatrix(vec))[0])
    mu_a = float(xgb_a.predict(xgb.DMatrix(vec))[0])
    return _prob_table(mu_h, mu_a, max_g)

###############################################################################
# 3.  LightGBM Poisson                                                         #
###############################################################################
lgb_params = dict(objective="poisson", metric="poisson", learning_rate=.05,
                  max_depth=5, num_leaves=31, subsample=.8, colsample_bytree=.8,
                  verbose=-1)
lgb_h = lgb.train(lgb_params,
                  lgb.Dataset(X_oh, matches.HomeGoals),
                  num_boost_round=600)
lgb_a = lgb.train(lgb_params,
                  lgb.Dataset(X_oh, matches.AwayGoals),
                  num_boost_round=600)

def predict_lgb(home, away, max_g=8):
    vec = _vectorize(home, away, X_oh.columns)
    mu_h = float(lgb_h.predict(vec)[0])
    mu_a = float(lgb_a.predict(vec)[0])
    return _prob_table(mu_h, mu_a, max_g)

###############################################################################
#  Shared helpers                                                              #
###############################################################################
def _vectorize(home, away, cols):
    v = pd.get_dummies(pd.DataFrame({"HomeTeam":[home], "AwayTeam":[away]}))
    v["home"] = 1
    return v.reindex(columns=cols, fill_value=0)

def _prob_table(mu_h, mu_a, max_g):
    pmf_h = poisson.pmf(np.arange(max_g+1)[:,None], mu=mu_h)
    pmf_a = poisson.pmf(np.arange(max_g+1)[None,:],  mu=mu_a)
    mat   = pmf_h*pmf_a
    hw,dw,aw = np.tril(mat,-1).sum(), np.diag(mat).sum(), np.triu(mat,1).sum()
    exp_h = int(round((mat.sum(1)*np.arange(mat.shape[0])).sum()))
    exp_a = int(round((mat.sum(0)*np.arange(mat.shape[1])).sum()))
    return dict(HG=exp_h, AG=exp_a, HW=hw, D=dw, AW=aw)

###############################################################################
#  Interactive widget                                                          #
###############################################################################
clubs = sorted(matches.HomeTeam.unique())
home_dd  = widgets.Dropdown(options=clubs, description="Home:")
away_dd  = widgets.Dropdown(options=clubs, description="Away:")
model_dd = widgets.ToggleButtons(
    options=[("Poisson GLM","pois"), ("XGBoost Poisson","xgb"), ("LightGBM Poisson","lgb")],
    description="Model:")
out = widgets.Output()
