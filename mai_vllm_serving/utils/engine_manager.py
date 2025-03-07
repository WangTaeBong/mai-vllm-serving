class EngineManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("엔진이 초기화되지 않았습니다")
        return cls._instance

    @classmethod
    def set_instance(cls, engine):
        cls._instance = engine
