from League.opponent import Opponent
import hockey.hockey_env as h_env

class HandcraftedOpponent(Opponent):
    def __init__(self, name: str, setup_name: str):
        self.name = name
        self.opponent = None
        if setup_name == "strong":
            self.opponent = h_env.BasicOpponent(weak=False)
        elif setup_name == "waek":
            self.opponent = h_env.BasicOpponent(weak=True)
        else:
            raise ValueError(f"Unknown setup name: {setup_name}")

    def act(self, obs):
        return self.opponent.act(obs)