from agent.state import StateCollector


class RimEnv:
    def __init__(self, max_step=50) -> None:
        self.current_state = StateCollector.current_state
        self.height = self.current_state.map_state.height
        self.width = self.current_state.map_state.width
        self.max_step = max_step
        self.ally_locs, self.enemy_locs, self.ally_cap, self.enemy_cap = (
            self.pawn_init()
        )

    def pawn_init(self):
        ally_locs, ally_cap = {}, {}
        enemy_locs, enemy_cap = {}, {}
        for label, pawn in self.current_state.pawn_states.items():
            if pawn.is_ally:
                ally_locs[label] = [pawn.loc[0], pawn.loc[1]]
                ally_cap[label] = pawn.is_incapable
            else:
                enemy_locs[label] = [pawn.loc[0], pawn.loc[1]]
                enemy_cap[label] = pawn.is_incapable
        return ally_locs, enemy_locs, ally_cap, enemy_cap
