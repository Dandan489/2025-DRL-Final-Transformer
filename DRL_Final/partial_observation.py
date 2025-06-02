import numpy as np

class ObservationMasker:
    def __init__(self, player_id: int = 1):
        self.player_id = player_id
        self.sight_radius_by_type = [0, 0, 5, 3, 3, 2, 2, 3]
        self.ownership_start = 10  # Player one-hot: index 10-12
        self.unit_type_start = 13  # Unit type one-hot: index 13-20

    def get_allied_unit_positions_and_types(self, observation):
        """Return list of (x, y, unit_type_index) for allied units."""
        h, w, _ = observation.shape
        player_onehot = np.zeros(3, dtype=int)
        player_onehot[self.player_id] = 1
        positions = []

        for i in range(h):
            for j in range(w):
                if np.all(observation[i, j, self.ownership_start:self.ownership_start+3] == player_onehot):
                    unit_type = observation[i, j, self.unit_type_start:self.unit_type_start+8]
                    if unit_type.sum() == 1:
                        type_index = np.argmax(unit_type)
                        positions.append((i, j, type_index))
        return positions

    def compute_visibility_mask(self, observation):
        """Compute the visibility mask based on unit positions and their sight radii."""
        h, w, _ = observation.shape
        mask = np.zeros((h, w), dtype=bool)
        units = self.get_allied_unit_positions_and_types(observation)

        for x, y, unit_type in units:
            radius = self.sight_radius_by_type[unit_type]
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if abs(dx) + abs(dy) <= radius:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            mask[nx, ny] = True
        return mask

    def mask_observation(self, observation: np.ndarray) -> np.ndarray:
        visibility = self.compute_visibility_mask(observation)
        return observation * visibility[:, :, None]

# Example:
# obs = np.random.randint(0, 2, size=(32, 32, 29), dtype=np.int32)
# masker = ObservationMasker(player_id=1)
# masked_obs = masker.mask_observation(obs)
