from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class ForegroundSettings:
    screen: str = 'blue'
    resolution: int = 2048
    exposure_ev: float = -2.
    strength: float = 1.
    edge_power: float = 4.
    despill: float = 0.
    device: str = 'cuda'
    precision: str = 'fp16'
    # Optional first original EXR. Frame zero maps to plate_first, explicitly.
    plate_path: str = ''
    plate_first: int = 0
    color_space: str = 'srgb'

    def validate(self):
        if self.screen not in ('blue', 'green') or self.resolution not in (1024, 1536, 2048):
            raise ValueError('Select a Blue/Green CorridorKey checkpoint and 1024, 1536 or 2048 resolution.')
        for name, low, high in [('exposure_ev', -6, 4), ('strength', 0, 1), ('edge_power', 1, 8), ('despill', 0, 1)]:
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Invalid CorridorKey {name}: expected {low}–{high}.')
        if self.device not in ('cuda', 'cpu') or self.precision not in ('fp16', 'fp32'):
            raise ValueError('Invalid CorridorKey device or precision.')
        if self.color_space not in ('srgb', 'linear709', 'acescg'):
            raise ValueError('Select sRGB, linear Rec.709 or ACEScg input color.')
        # Input color is checked against the actual project in prepare(): native
        # EXR projects can supply linear RGB without an external plate override.
        if self.plate_path and not self.plate_path.lower().endswith('.exr'):
            raise ValueError('The original plate override must be the first numbered RGB EXR.')
        if self.plate_first < 0:
            raise ValueError('The first plate frame must be nonnegative.')
        return self

    @classmethod
    def from_dict(cls, data):
        keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in keys})

    def inference(self):
        # Strength/despill are inexpensive RGB operations; editing them reuses inference.
        return {k: v for k, v in asdict(self).items() if k not in ('strength', 'edge_power', 'despill', 'plate_path')}


PRESETS = {
    'Blue sky / hair (tested)': dict(screen='blue', resolution=2048, exposure_ev=-2., strength=1., edge_power=4., despill=0.),
    'Blue screen / neutral exposure': dict(screen='blue', resolution=2048, exposure_ev=0., strength=1., edge_power=4., despill=0.),
    'Green screen / neutral exposure': dict(screen='green', resolution=2048, exposure_ev=0., strength=1., edge_power=4., despill=0.),
    'Fast blue preview': dict(screen='blue', resolution=1024, exposure_ev=-2., strength=1., edge_power=4., despill=0.),
}
