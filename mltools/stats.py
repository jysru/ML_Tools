from abc import ABC, abstractmethod
import numpy as np



class AbstractStatsNormalizer(ABC):
    
    def __init__(self, data: np.ndarray = None, stats: dict = None) -> None:
        if data is not None:
            self._compute_stats(data=data)
        if stats is not None:
            self._assign_stats(stats_dict=stats)
    
    def _compute_stats(self, data) -> None:
        self._mean = np.mean(data)
        self._std = np.std(data)
        self._median = np.median(data)
        self._min = np.min(data)
        self._max = np.max(data)
        self._perc10 = np.percentile(data, 10)
        self._perc25 = np.percentile(data, 25)
        self._perc75 = np.percentile(data, 75)
        self._perc90 = np.percentile(data, 90)
        
    def _assign_stats(self, stats_dict: dict) -> None:
        self._mean = stats_dict['mean']
        self._std = stats_dict['std']
        self._median = stats_dict['median']
        self._min = stats_dict['min']
        self._max = stats_dict['max']
        self._perc10 = stats_dict['perc10']
        self._perc25 = stats_dict['perc25']
        self._perc75 = stats_dict['perc75']
        self._perc90 = stats_dict['perc90']
        
    def export_stats(self) -> dict:
        return dict(
            mean = self._mean,
            std = self._std,
            median = self._median,
            min = self._min,
            max = self._max,
            perc10 = self._perc10,
            perc25 = self._perc25,
            perc75 = self._perc75,
            perc90 = self._perc90,
        )
        
    @abstractmethod
    def normalize(self, data: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        pass


class GaussianNormalizer(AbstractStatsNormalizer):
    
    def __init__(self, data: np.ndarray = None, stats: dict = None) -> None:
        super().__init__(data, stats)
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self._mean) / self._std
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * self._std + self._mean
    
    
class IQRNormalizer(AbstractStatsNormalizer):
    
    def __init__(self, data: np.ndarray = None, stats: dict = None) -> None:
        super().__init__(data, stats)
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self._median) / (self._perc75 - self._perc25)
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * (self._perc75 - self._perc25) + self._median
    
    
class I90RNormalizer(AbstractStatsNormalizer):
    
    def __init__(self, data: np.ndarray = None, stats: dict = None) -> None:
        super().__init__(data, stats)
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self._median) / (self._perc90 - self._perc10)
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * (self._perc90 - self._perc10) + self._median
    
    
class MinMaxNormalizer(AbstractStatsNormalizer):
    
    def __init__(self, data: np.ndarray = None, stats: dict = None) -> None:
        super().__init__(data, stats)
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self._median) / (self._max - self._min)
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * (self._max - self._min) + self._median
    
    
class MaxNormalizer(AbstractStatsNormalizer):
    
    def __init__(self, data: np.ndarray = None, stats: dict = None) -> None:
        super().__init__(data, stats)
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return data / self._max
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * self._max
    
    
class NoNormalizer(AbstractStatsNormalizer):
    
    def __init__(self, data: np.ndarray = None, stats: dict = None) -> None:
        super().__init__(data, stats)
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return data
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data
    
    
class GlobalNormalizer(AbstractStatsNormalizer):
    
    def __init__(self, data: np.ndarray = None, stats: dict = None, normalizer_class: AbstractStatsNormalizer = MinMaxNormalizer) -> None:
        super().__init__(data, stats)
        self._normalizer = normalizer_class(data=None, stats=self.export_stats())
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return self._normalizer.normalize(data)

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return self._normalizer.denormalize(data)
    
    
class PerVariableNormalizer(AbstractStatsNormalizer):
    """
        (De)Normalizes data based on individual variable statistics
        
        Assumes input data individual variables are placed on the first axis
    """
    
    def __init__(self, data: np.ndarray = None, stats: dict = None, normalizer_class: AbstractStatsNormalizer = MinMaxNormalizer) -> None:
        # super().__init__(data, stats)
        self._normalizer_class = normalizer_class
        self._init_normalizers(data=data, stats=stats)
    
    def _init_normalizers(self, data: np.ndarray = None, stats: list[dict] = None) -> None:
        if data is not None:
            self._init_normalizers_from_data(data=data)
        if stats is not None:
            self._init_normalizers_from_stats(stats=stats)
            
    def _init_normalizers_from_data(self, data: np.ndarray) -> None:
        self._normalizers = []
        for i in range(data.shape[1]):
            self._normalizers.append(
                self._normalizer_class(data=data[..., i], stats=None)
            )
            
    def _init_normalizers_from_stats(self, stats: list[dict]) -> None:
        self._normalizers = []
        for i in range(len(stats)):
            self._normalizers.append(
                self._normalizer_class(data=None, stats=stats[i])
            )
            
    def normalize(self, data: np.ndarray) -> np.ndarray:
        data = np.copy(data)
        for i in range(len(self._normalizers)):
            data[..., i] = self._normalizers[i].normalize(data[..., i])
        return data

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        data = np.copy(data)
        for i in range(len(self._normalizers)):
            data[..., i] = self._normalizers[i].denormalize(data[..., i])
        return data
    
    def export_stats(self) -> list[dict]:
        stats_dicts = []
        for i in range(len(self._normalizers)):
            stats_dicts.append(
                self._normalizers[i].export_stats()
            )
        return stats_dicts
