"""Application-facing segmentation interface and backend lifecycle."""
from sammie import core
from sammie.settings_manager import get_settings_manager


class SamManager(core.CallbackMixin):
    def __init__(self):
        super().__init__()
        self.backend = None
        self._propagated = self._deduplicated = False

    def __getattr__(self, name):
        backend = self.__dict__.get('backend')
        if backend is not None:
            return getattr(backend, name)
        if name in ('loaded_model_name', 'predictor', 'inference_state'):
            return None
        if name.startswith('supports_'):
            return False
        raise RuntimeError('Load a segmentation model before editing or tracking.')

    @property
    def propagated(self):
        return self.backend.propagated if self.backend else self._propagated

    @propagated.setter
    def propagated(self, value):
        self._propagated = value
        if self.backend:
            self.backend.propagated = value

    @property
    def deduplicated(self):
        return self.backend.deduplicated if self.backend else self._deduplicated

    @deduplicated.setter
    def deduplicated(self, value):
        self._deduplicated = value
        if self.backend:
            self.backend.deduplicated = value

    def load_segmentation_model(self, model=None, parent_window=None):
        model = model or get_settings_manager().get_session_setting('sam_model', 'Base')
        from .sam2 import Sam2Backend
        from .sam3 import Sam3Backend
        factory = {'Base': Sam2Backend, 'Large': Sam2Backend, 'Efficient': Sam2Backend,
                   'Anime': Sam2Backend, 'SAM 3': Sam3Backend, 'SAM 3.1': Sam3Backend}
        if model not in factory:
            raise ValueError(f'Unknown segmentation model: {model}')
        self.unload_segmentation_model()
        backend = factory[model]()
        backend.add_callback(self._notify)
        try:
            if not backend.load_segmentation_model(model, parent_window):
                backend.unload_segmentation_model()
                return False
        except Exception:
            backend.unload_segmentation_model()
            raise
        self.backend = backend
        return True

    def unload_segmentation_model(self):
        if self.backend:
            self.backend.unload_segmentation_model()
        self.backend = None

    def clear_tracking(self):
        if self.backend:
            self.backend.clear_tracking()

    def initialize_predictor(self):
        if self.backend:
            self.backend.initialize_predictor()

    def reset_session(self):
        if self.backend:
            self.backend.reset_session()
