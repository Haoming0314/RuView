"""WiFi-DensePose 系统的相位净化模块，采用 TDD 开发方式。"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from scipy import signal


class PhaseSanitizationError(Exception):
    """相位净化过程中抛出的异常。"""
    pass


class PhaseSanitizer:
    """对 CSI 信号中的相位数据进行净化，以支持可靠处理。"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """初始化相位净化器。
        
        Args:
            config: 配置字典
            logger: 可选的日志记录器实例
            
        Raises:
            ValueError: 当配置无效时抛出
        """
        self._validate_config(config)
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 处理参数
        self.unwrapping_method = config['unwrapping_method']
        self.outlier_threshold = config['outlier_threshold']
        self.smoothing_window = config['smoothing_window']
        
        # 带默认值的可选参数
        self.enable_outlier_removal = config.get('enable_outlier_removal', True)
        self.enable_smoothing = config.get('enable_smoothing', True)
        self.enable_noise_filtering = config.get('enable_noise_filtering', False)
        self.noise_threshold = config.get('noise_threshold', 0.05)
        self.phase_range = config.get('phase_range', (-np.pi, np.pi))
        
        # 统计信息跟踪
        self._total_processed = 0
        self._outliers_removed = 0
        self._sanitization_errors = 0
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """校验配置参数。
        
        Args:
            config: 待校验的配置
            
        Raises:
            ValueError: 当配置无效时抛出
        """
        required_fields = ['unwrapping_method', 'outlier_threshold', 'smoothing_window']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")
        
        # 校验相位展开方法
        valid_methods = ['numpy', 'scipy', 'custom']
        if config['unwrapping_method'] not in valid_methods:
            raise ValueError(f"Invalid unwrapping method: {config['unwrapping_method']}. Must be one of {valid_methods}")
        
        # 校验阈值参数
        if config['outlier_threshold'] <= 0:
            raise ValueError("outlier_threshold must be positive")
        
        if config['smoothing_window'] <= 0:
            raise ValueError("smoothing_window must be positive")
    
    def unwrap_phase(self, phase_data: np.ndarray) -> np.ndarray:
        """对相位数据进行展开，以消除不连续跳变。
        
        Args:
            phase_data: 已包裹的相位数据（二维数组）
            
        Returns:
            展开后的相位数据
            
        Raises:
            PhaseSanitizationError: 当相位展开失败时抛出
        """
        try:
            if self.unwrapping_method == 'numpy':
                return self._unwrap_numpy(phase_data)
            elif self.unwrapping_method == 'scipy':
                return self._unwrap_scipy(phase_data)
            elif self.unwrapping_method == 'custom':
                return self._unwrap_custom(phase_data)
            else:
                raise ValueError(f"Unknown unwrapping method: {self.unwrapping_method}")
                
        except Exception as e:
            raise PhaseSanitizationError(f"Failed to unwrap phase: {e}")
    
    def _unwrap_numpy(self, phase_data: np.ndarray) -> np.ndarray:
        """使用 numpy 的 unwrap 函数进行相位展开。"""
        if phase_data.size == 0:
            raise ValueError("Cannot unwrap empty phase data")
        return np.unwrap(phase_data, axis=1)
    
    def _unwrap_scipy(self, phase_data: np.ndarray) -> np.ndarray:
        """使用 scipy 的 unwrap 函数进行相位展开。"""
        if phase_data.size == 0:
            raise ValueError("Cannot unwrap empty phase data")
        return np.unwrap(phase_data, axis=1)
    
    def _unwrap_custom(self, phase_data: np.ndarray) -> np.ndarray:
        """使用自定义算法进行相位展开。"""
        if phase_data.size == 0:
            raise ValueError("Cannot unwrap empty phase data")
        # 简单的自定义相位展开算法
        unwrapped = phase_data.copy()
        for i in range(phase_data.shape[0]):
            unwrapped[i, :] = np.unwrap(phase_data[i, :])
        return unwrapped
    
    def remove_outliers(self, phase_data: np.ndarray) -> np.ndarray:
        """移除相位数据中的离群值。
        
        Args:
            phase_data: 相位数据（二维数组）
            
        Returns:
            已去除离群值的相位数据
            
        Raises:
            PhaseSanitizationError: 当离群值移除失败时抛出
        """
        if not self.enable_outlier_removal:
            return phase_data
        
        try:
            # 检测离群值
            outlier_mask = self._detect_outliers(phase_data)
            
            # 对离群值进行插值
            clean_data = self._interpolate_outliers(phase_data, outlier_mask)
            
            return clean_data
            
        except Exception as e:
            raise PhaseSanitizationError(f"Failed to remove outliers: {e}")
    
    def _detect_outliers(self, phase_data: np.ndarray) -> np.ndarray:
        """使用统计方法检测离群值。"""
        # 使用 Z-score 方法检测离群值
        z_scores = np.abs((phase_data - np.mean(phase_data, axis=1, keepdims=True)) / 
                         (np.std(phase_data, axis=1, keepdims=True) + 1e-8))
        outlier_mask = z_scores > self.outlier_threshold
        
        # 更新统计信息
        self._outliers_removed += np.sum(outlier_mask)
        
        return outlier_mask
    
    def _interpolate_outliers(self, phase_data: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        """对离群值进行插值。"""
        clean_data = phase_data.copy()
        
        for i in range(phase_data.shape[0]):
            outliers = outlier_mask[i, :]
            if np.any(outliers):
                # 对离群值执行线性插值
                valid_indices = np.where(~outliers)[0]
                outlier_indices = np.where(outliers)[0]
                
                if len(valid_indices) > 1:
                    clean_data[i, outlier_indices] = np.interp(
                        outlier_indices, valid_indices, phase_data[i, valid_indices]
                    )
        
        return clean_data
    
    def smooth_phase(self, phase_data: np.ndarray) -> np.ndarray:
        """对相位数据进行平滑以降低噪声。
        
        Args:
            phase_data: 相位数据（二维数组）
            
        Returns:
            平滑后的相位数据
            
        Raises:
            PhaseSanitizationError: 当平滑处理失败时抛出
        """
        if not self.enable_smoothing:
            return phase_data
        
        try:
            smoothed_data = self._apply_moving_average(phase_data, self.smoothing_window)
            return smoothed_data
            
        except Exception as e:
            raise PhaseSanitizationError(f"Failed to smooth phase: {e}")
    
    def _apply_moving_average(self, phase_data: np.ndarray, window_size: int) -> np.ndarray:
        """应用滑动平均平滑。"""
        smoothed_data = phase_data.copy()
        
        # 保证窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        
        for i in range(phase_data.shape[0]):
            for j in range(half_window, phase_data.shape[1] - half_window):
                start_idx = j - half_window
                end_idx = j + half_window + 1
                smoothed_data[i, j] = np.mean(phase_data[i, start_idx:end_idx])
        
        return smoothed_data
    
    def filter_noise(self, phase_data: np.ndarray) -> np.ndarray:
        """过滤相位数据中的噪声。
        
        Args:
            phase_data: 相位数据（二维数组）
            
        Returns:
            滤波后的相位数据
            
        Raises:
            PhaseSanitizationError: 当噪声滤波失败时抛出
        """
        if not self.enable_noise_filtering:
            return phase_data
        
        try:
            filtered_data = self._apply_low_pass_filter(phase_data, self.noise_threshold)
            return filtered_data
            
        except Exception as e:
            raise PhaseSanitizationError(f"Failed to filter noise: {e}")
    
    def _apply_low_pass_filter(self, phase_data: np.ndarray, threshold: float) -> np.ndarray:
        """应用低通滤波器以去除高频噪声。"""
        filtered_data = phase_data.copy()
        
        # 检查数据长度是否足以进行滤波
        min_filter_length = 18  # 四阶 filtfilt 所需的最小长度
        if phase_data.shape[1] < min_filter_length:
            # 对过小的数组跳过滤波
            return filtered_data
        
        # 应用 Butterworth 低通滤波器
        nyquist = 0.5
        cutoff = threshold * nyquist
        
        # 设计滤波器
        b, a = signal.butter(4, cutoff, btype='low')
        
        # 对每根天线分别应用滤波
        for i in range(phase_data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, phase_data[i, :])
        
        return filtered_data
    
    def sanitize_phase(self, phase_data: np.ndarray) -> np.ndarray:
        """通过完整流水线对相位数据进行净化。
        
        Args:
            phase_data: 原始相位数据（二维数组）
            
        Returns:
            净化后的相位数据
            
        Raises:
            PhaseSanitizationError: 当净化失败时抛出
        """
        try:
            self._total_processed += 1
            
            # 校验输入数据
            self.validate_phase_data(phase_data)
            
            # 执行完整的净化流水线
            sanitized_data = self.unwrap_phase(phase_data)
            sanitized_data = self.remove_outliers(sanitized_data)
            sanitized_data = self.smooth_phase(sanitized_data)
            sanitized_data = self.filter_noise(sanitized_data)
            
            return sanitized_data
            
        except PhaseSanitizationError:
            self._sanitization_errors += 1
            raise
        except Exception as e:
            self._sanitization_errors += 1
            raise PhaseSanitizationError(f"Sanitization pipeline failed: {e}")
    
    def validate_phase_data(self, phase_data: np.ndarray) -> bool:
        """校验相位数据的格式和值范围。
        
        Args:
            phase_data: 待校验的相位数据
            
        Returns:
            校验通过时返回 True
            
        Raises:
            PhaseSanitizationError: 当校验失败时抛出
        """
        # 检查数据是否为二维
        if phase_data.ndim != 2:
            raise PhaseSanitizationError("Phase data must be 2D array")
        
        # 检查数据是否为空
        if phase_data.size == 0:
            raise PhaseSanitizationError("Phase data cannot be empty")
        
        # 检查取值是否位于合法范围内
        min_val, max_val = self.phase_range
        if np.any(phase_data < min_val) or np.any(phase_data > max_val):
            raise PhaseSanitizationError(f"Phase values outside valid range [{min_val}, {max_val}]")
        
        return True
    
    def get_sanitization_statistics(self) -> Dict[str, Any]:
        """获取净化统计信息。
        
        Returns:
            包含净化统计信息的字典
        """
        outlier_rate = self._outliers_removed / self._total_processed if self._total_processed > 0 else 0
        error_rate = self._sanitization_errors / self._total_processed if self._total_processed > 0 else 0
        
        return {
            'total_processed': self._total_processed,
            'outliers_removed': self._outliers_removed,
            'sanitization_errors': self._sanitization_errors,
            'outlier_rate': outlier_rate,
            'error_rate': error_rate
        }
    
    def reset_statistics(self) -> None:
        """重置净化统计信息。"""
        self._total_processed = 0
        self._outliers_removed = 0
        self._sanitization_errors = 0
