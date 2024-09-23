from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


    
class AbstractTransformation(ABC):
    
    @abstractmethod
    def forward(self, data: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, data: np.ndarray) -> np.ndarray:
        pass
    
    def __call__(self, data: np.ndarray, invert: bool = False) -> np.ndarray:
        if invert:
            return self.backward(data)
        else:
            return self.forward(data)
        
    
class Intensity2Amplitude(AbstractTransformation):
    
    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.sqrt(data)
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        return np.square(data)
    
    def __call__(self, data: np.ndarray, invert: bool = False) -> np.ndarray:
        return super().__call__(data, invert)
    
    
class Amplitude2Intensity(AbstractTransformation):
    use_tensorflow: bool = False
    
    def __init__(self, use_tensorflow: bool = None) -> None:
        super().__init__()
        if use_tensorflow is not None:
            self.use_tensorflow = use_tensorflow
    
    def forward(self, data: np.ndarray) -> np.ndarray:
        if not self.use_tensorflow:
            return self._forward_numpy(data)
        else:
            return self._forward_tensorflow(data)
    
    def _forward_numpy(self, data: np.ndarray) -> np.ndarray:
        return np.square(data)
    
    def _forward_tensorflow(self, data: tf.Tensor) -> tf.Tensor:
        return tf.math.square(data)
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        if not self.use_tensorflow:
            return self._backward_numpy(data)
        else:
            return self._backward_tensorflow(data)
    
    def _backward_numpy(self, data: np.ndarray) -> np.ndarray:
        return np.sqrt(data)
    
    def _backward_tensorflow(self, data: tf.Tensor) -> tf.Tensor:
        return tf.math.sqrt(data)
    
    def __call__(self, data: np.ndarray, invert: bool = False) -> np.ndarray:
        return super().__call__(data, invert)
    
    
class Linear2Decibels(AbstractTransformation):
    epsilon: float = 1e-9
    use_tensorflow: bool = False
    
    def __init__(self, epsilon: float = None, use_tensorflow: bool = None) -> None:
        super().__init__()
        if epsilon is not None:
            self.epsilon = epsilon
        if use_tensorflow is not None:
            self.use_tensorflow = use_tensorflow
    
    def forward(self, data: np.ndarray) -> np.ndarray:
        if not self.use_tensorflow:
            return self._forward_numpy(data)
        else:
            return self._forward_tensorflow(data)
    
    def _forward_numpy(self, data: np.ndarray) -> np.ndarray:
        return 10 * np.log10(data + self.epsilon)
    
    def _forward_tensorflow(self, data: tf.Tensor) -> tf.Tensor:
        return 10 * tf.math.log(data + self.epsilon) / tf.math.log(10)
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        if not self.use_tensorflow:
            return self._backward_numpy(data)
        else:
            return self._backward_tensorflow(data)
    
    def _backward_numpy(self, data: np.ndarray) -> np.ndarray:
        return np.power(10, (data + self.epsilon) / 10)
    
    def _backward_tensorflow(self, data: tf.Tensor) -> tf.Tensor:
        return tf.math.pow(10, (data + self.epsilon) / 10)
    
    def __call__(self, data: np.ndarray, invert: bool = False) -> np.ndarray:
        return super().__call__(data, invert)
    
    
    
class DynamicRangeClipper(AbstractTransformation):
    
    def __init__(self, dynamic_range_db: float = 30) -> None:
        super().__init__()
        self.dynamic_range_decibels: float = dynamic_range_db
        
    def forward(self, data: np.ndarray) -> np.ndarray:
        data = Linear2Decibels().forward(data)
        data[data <= (np.max(data) - np.abs(self.dynamic_range_decibels))] = 0
        data = Linear2Decibels().backward(data)
        
    def backward(self) -> None:
        return None
