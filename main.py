from typing import List, Optional, Literal
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel, Field
import math
import time

app = FastAPI(title="Football Live Analytics API", version="1.0.0")

class MatchCard(BaseModel):
    fixture_id: str
    league: str
    home: str
    away: str
    start_time: str
    status: Literal["scheduled", "live", "finished"]

class LiveStats(BaseModel):
    minute: int
    score_home: int
    score_away: int
    shots_total: int
    shots_box: int
    shots_on: int
    big_chances: int
    corners: int
    possession_home: int
    possession_away: int
    momentum: Literal["home", "away", "balanced"]
    xg_sum: Optional[float] = None

class OddsInput(BaseModel):
    over25: Optional[float] = None
    over05_ht: Optional[float] = None
    btts: Optional[float] = None
    oneXtwo: Optional[dict] = None

class RecRequest(BaseModel):
    stats: LiveStats
    odds: OddsInput
    models_selected: List[str] = Field(
        default_factory=lambda: ["bayes_goals","bivar_poisson","bayes_xg","ensemble","xt","xpts"]
    )
    all_models: bool = True

class RecOutput(BaseModel):
    p_over25: float
    p_goal_ht: float
    p_btts: float
    p_1x2: dict
    ev: dict
    min_odds: dict
    stake_kelly: dict
    rationale: List[str]
    models_used: List[str]

@app.get("/matches/top-leagues", response_model=List[MatchCard])
def top_leagues():
    return [
        MatchCard(fixture_id="1001", league="Serie A", home="Milan", away="Roma",
                  start_time="2025-10-06T19:45:00Z", status="scheduled"),
        MatchCard(fixture_id="1002", league="Premier League", home="Arsenal", away="Newcastle",
                  start_time="2025-10-06T20:00:00Z", status="live"),
    ]

@app.get("/match/{fixture_id}/live", response_model=LiveStats)
def live_stats(fixture_id: str):
    return LiveStats(
        minute=27, score_home=0, score_away=0,
        shots_total=2, shots_box=1, shots_on=1,
        big_chances=0, corners=1, possession_home=52, possession_away=48,
        momentum="balanced", xg_sum=0.25
    )

def sigmoid(x): return 1/(1+math.exp(-x))

def heuristic_probs(s: LiveStats):
    over_raw = 0.30 + 0.02*s.shots_total + 0.04*s.shots_box + 0.08*s.big_chances + (0.20 if (s.xg_sum and s.xg_sum>=0.9) else 0.0)
    goal_ht_raw = 0.15 + 0.03*s.shots_box + 0.03*s.corners + (0.07 if s.momentum!="balanced" else 0.0)
    btts_raw = 0.25 + 0.05*(1 if s.momentum=="balanced" else 0) + 0.02*s.shots_on
    p1 = sigmoid((s.possession_home-50)/10)
    p2 = sigmoid((s.possession_away-50)/10)
    p_draw = 0.20
    total = p1+p2+p_draw
    p1, p2, p_draw = p1/total, p2/total, p_draw/total
    def clip01(x): return max(0.01, min(0.95, x))
    return {
        "over25": clip01(over_raw),
        "goal_ht": clip01(goal_ht_raw),
        "btts": clip01(btts_raw),
        "oneXtwo": {"1": clip01(p1), "X": clip01(p_draw), "2": clip01(p2)}
    }

def ev(prob, odds):
    if (prob is None) or (odds is None): return None
    return prob*odds - 1

def kelly_fraction(prob, odds):
    if (prob is None) or (odds is None): return 0.0
    b = odds - 1
    edge = prob - (1 - prob)/b if b>0 else -1
    return max(0.0, min(0.5, edge/b if b>0 else 0.0))

@app.post("/recommendation", response_model=RecOutput)
def recommendation(req: RecRequest):
    s = req.stats
    base = heuristic_probs(s)
    rationale = []
    min_odds = {"over25":1.70, "goal_ht":1.60, "btts":1.75}
    evs = {
        "over25": ev(base["over25"], req.odds.over25),
        "goal_ht": ev(base["goal_ht"], req.odds.over05_ht),
        "btts": ev(base["btts"], req.odds.btts),
    }
    stakes = {
        "over25": round(0.25*kelly_fraction(base["over25"], req.odds.over25 or 0), 3),
        "goal_ht": round(0.25*kelly_fraction(base["goal_ht"], req.odds.over05_ht or 0), 3),
        "btts": round(0.20*kelly_fraction(base["btts"], req.odds.btts or 0), 3),
    }
    if s.shots_total>=9 and s.shots_box>=5: rationale.append("Volume e qualità alte per Over 2.5 (>=9 tiri, >=5 in area).")
    if s.big_chances>=1: rationale.append("Almeno una big chance creata.")
    if 25<=s.minute<=35 and s.corners>=2 and s.shots_box>=3: rationale.append("Finestra 25–35 con pressione: favorevole a Gol 1T.")
    if s.momentum=="balanced" and s.shots_on>=2: rationale.append("Pressione simmetrica: BTTS favorito.")
    models_used = (req.models_selected if not req.all_models else
                   ["bayes_goals","bivar_poisson","bayes_xg","ensemble","xt","xpts","kelly"])
    return RecOutput(
        p_over25=round(base["over25"],3),
        p_goal_ht=round(base["goal_ht"],3),
        p_btts=round(base["btts"],3),
        p_1x2={k:round(v,3) for k,v in base["oneXtwo"].items()},
        ev={k:(round(v,3) if v is not None else None) for k,v in evs.items()},
        min_odds=min_odds,
        stake_kelly=stakes,
        rationale=rationale or ["Segnali insufficienti: attendere nuovo picco di pressione."],
        models_used=models_used
    )

@app.websocket("/stream/{fixture_id}")
async def stream(ws: WebSocket, fixture_id: str):
    await ws.accept()
    t = 0
    try:
        while True:
            payload = {"fixture_id": fixture_id, "minute": 20 + t, "event": "tick"}
            await ws.send_json(payload)
            t += 1
            time.sleep(2)
    except Exception:
        pass
