"""WiFi-DensePose 系统的 CSI 数据处理器，采用 TDD 开发方式。"""

import asyncio
import itertools
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque
import scipy.signal
import scipy.fft

try:
    from ..hardware.csi_extractor import CSIData
except ImportError:
    # 为测试场景处理导入路径
    from src.hardware.csi_extractor import CSIData


class CSIProcessingError(Exception):
    """CSI 处理过程中抛出的异常。"""
    pass


@dataclass
class CSIFeatures:
    """提取后的 CSI 特征数据结构。"""
    amplitude_mean: np.ndarray
    amplitude_variance: np.ndarray
    phase_difference: np.ndarray
    correlation_matrix: np.ndarray
    doppler_shift: np.ndarray
    power_spectral_density: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class HumanDetectionResult:
    """人体检测结果数据结构。"""
    human_detected: bool
    confidence: float
    motion_score: float
    timestamp: datetime
    features: CSIFeatures
    metadata: Dict[str, Any]


class CSIProcessor:
    """对 CSI 数据进行处理，用于人体检测和姿态估计。"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """初始化 CSI 处理器。
        
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
        self.sampling_rate = config['sampling_rate']
        self.window_size = config['window_size']
        self.overlap = config['overlap']
        self.noise_threshold = config['noise_threshold']
        self.human_detection_threshold = config.get('human_detection_threshold', 0.8)
        self.smoothing_factor = config.get('smoothing_factor', 0.9)
        self.max_history_size = config.get('max_history_size', 500)
        
        # 特征提取开关
        self.enable_preprocessing = config.get('enable_preprocessing', True)
        self.enable_feature_extraction = config.get('enable_feature_extraction', True)
        self.enable_human_detection = config.get('enable_human_detection', True)
        
        # 处理状态
        self.csi_history = deque(maxlen=self.max_history_size)
        self.previous_detection_confidence = 0.0

        # 多普勒缓存：预先计算每帧的平均相位，以便 O(1) 追加
        self._phase_cache = deque(maxlen=self.max_history_size)
        self._doppler_window = min(config.get('doppler_window', 64), self.max_history_size)
        
        # 统计信息跟踪
        self._total_processed = 0
        self._processing_errors = 0
        self._human_detections = 0
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """校验配置参数。
        
        Args:
            config: 待校验的配置
            
        Raises:
            ValueError: 当配置无效时抛出
        """
        required_fields = ['sampling_rate', 'window_size', 'overlap', 'noise_threshold']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")
        
        if config['sampling_rate'] <= 0:
            raise ValueError("sampling_rate must be positive")
        
        if config['window_size'] <= 0:
            raise ValueError("window_size must be positive")
        
        if not 0 <= config['overlap'] < 1:
            raise ValueError("overlap must be between 0 and 1")
    
    def preprocess_csi_data(self, csi_data: CSIData) -> CSIData:
        """对 CSI 数据进行预处理，以便提取特征。
        
        Args:
            csi_data: 原始 CSI 数据
            
        Returns:
            预处理后的 CSI 数据
            
        Raises:
            CSIProcessingError: 当预处理失败时抛出
        """
        if not self.enable_preprocessing:
            return csi_data
        
        try:
            # 去除信号噪声
            cleaned_data = self._remove_noise(csi_data)
            
            # 应用窗函数
            windowed_data = self._apply_windowing(cleaned_data)
            
            # 对幅度值进行归一化
            normalized_data = self._normalize_amplitude(windowed_data)
            
            return normalized_data
            
        except Exception as e:
            raise CSIProcessingError(f"Failed to preprocess CSI data: {e}")
    
    def extract_features(self, csi_data: CSIData) -> Optional[CSIFeatures]:
        """从 CSI 数据中提取特征。
        
        Args:
            csi_data: 预处理后的 CSI 数据
            
        Returns:
            提取出的特征；若功能关闭则返回 None
            
        Raises:
            CSIProcessingError: 当特征提取失败时抛出
        """
        if not self.enable_feature_extraction:
            return None
        
        try:
            # 提取基于幅度的特征
            amplitude_mean, amplitude_variance = self._extract_amplitude_features(csi_data)
            
            # 提取基于相位的特征
            phase_difference = self._extract_phase_features(csi_data)
            
            # 提取相关性特征
            correlation_matrix = self._extract_correlation_features(csi_data)
            
            # 提取多普勒和频域特征
            doppler_shift, power_spectral_density = self._extract_doppler_features(csi_data)
            
            return CSIFeatures(
                amplitude_mean=amplitude_mean,
                amplitude_variance=amplitude_variance,
                phase_difference=phase_difference,
                correlation_matrix=correlation_matrix,
                doppler_shift=doppler_shift,
                power_spectral_density=power_spectral_density,
                timestamp=datetime.now(timezone.utc),
                metadata={'processing_params': self.config}
            )
            
        except Exception as e:
            raise CSIProcessingError(f"Failed to extract features: {e}")
    
    def detect_human_presence(self, features: CSIFeatures) -> Optional[HumanDetectionResult]:
        """根据 CSI 特征检测人体存在。
        
        Args:
            features: 提取出的 CSI 特征
            
        Returns:
            检测结果；若功能关闭则返回 None
            
        Raises:
            CSIProcessingError: 当检测失败时抛出
        """
        if not self.enable_human_detection:
            return None
        
        try:
            # 分析运动模式
            motion_score = self._analyze_motion_patterns(features)
            
            # 计算检测置信度
            raw_confidence = self._calculate_detection_confidence(features, motion_score)
            
            # 应用时间平滑
            smoothed_confidence = self._apply_temporal_smoothing(raw_confidence)
            
            # 判断是否检测到人体
            human_detected = smoothed_confidence >= self.human_detection_threshold
            
            if human_detected:
                self._human_detections += 1
            
            return HumanDetectionResult(
                human_detected=human_detected,
                confidence=smoothed_confidence,
                motion_score=motion_score,
                timestamp=datetime.now(timezone.utc),
                features=features,
                metadata={'threshold': self.human_detection_threshold}
            )
            
        except Exception as e:
            raise CSIProcessingError(f"Failed to detect human presence: {e}")
    
    async def process_csi_data(self, csi_data: CSIData) -> HumanDetectionResult:
        """通过完整流水线处理 CSI 数据。
        
        Args:
            csi_data: 原始 CSI 数据
            
        Returns:
            人体检测结果
            
        Raises:
            CSIProcessingError: 当处理失败时抛出
        """
        try:
            self._total_processed += 1
            
            # 预处理数据
            preprocessed_data = self.preprocess_csi_data(csi_data)
            
            # 提取特征
            features = self.extract_features(preprocessed_data)
            
            # 检测人体存在
            detection_result = self.detect_human_presence(features)
            
            # 加入历史记录
            self.add_to_history(csi_data)
            
            return detection_result
            
        except Exception as e:
            self._processing_errors += 1
            raise CSIProcessingError(f"Pipeline processing failed: {e}")
    
    def add_to_history(self, csi_data: CSIData) -> None:
        """将 CSI 数据加入处理历史记录。

        Args:
            csi_data: 要加入历史记录的 CSI 数据
        """
        self.csi_history.append(csi_data)
        # 缓存平均相位，用于快速提取多普勒特征
        if csi_data.phase.ndim == 2:
            self._phase_cache.append(np.mean(csi_data.phase, axis=0))
        else:
            self._phase_cache.append(csi_data.phase.flatten())
    
    def clear_history(self) -> None:
        """清空 CSI 数据历史记录。"""
        self.csi_history.clear()
        self._phase_cache.clear()
    
    def get_recent_history(self, count: int) -> List[CSIData]:
        """获取最近的 CSI 历史数据。
        
        Args:
            count: 要返回的最近记录数量
            
        Returns:
            最近 CSI 数据记录列表
        """
        if count >= len(self.csi_history):
            return list(self.csi_history)
        else:
            start = len(self.csi_history) - count
            return list(itertools.islice(self.csi_history, start, len(self.csi_history)))
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息。
        
        Returns:
            包含处理统计信息的字典
        """
        error_rate = self._processing_errors / self._total_processed if self._total_processed > 0 else 0
        detection_rate = self._human_detections / self._total_processed if self._total_processed > 0 else 0
        
        return {
            'total_processed': self._total_processed,
            'processing_errors': self._processing_errors,
            'human_detections': self._human_detections,
            'error_rate': error_rate,
            'detection_rate': detection_rate,
            'history_size': len(self.csi_history)
        }
    
    def reset_statistics(self) -> None:
        """重置处理统计信息。"""
        self._total_processed = 0
        self._processing_errors = 0
        self._human_detections = 0
    
    # 私有处理方法
    def _remove_noise(self, csi_data: CSIData) -> CSIData:
        """从 CSI 数据中去除噪声。"""
        # 根据阈值应用噪声过滤
        amplitude_db = 20 * np.log10(np.abs(csi_data.amplitude) + 1e-12)
        noise_mask = amplitude_db > self.noise_threshold
        
        filtered_amplitude = csi_data.amplitude.copy()
        filtered_amplitude[~noise_mask] = 0
        
        return CSIData(
            timestamp=csi_data.timestamp,
            amplitude=filtered_amplitude,
            phase=csi_data.phase,
            frequency=csi_data.frequency,
            bandwidth=csi_data.bandwidth,
            num_subcarriers=csi_data.num_subcarriers,
            num_antennas=csi_data.num_antennas,
            snr=csi_data.snr,
            metadata={**csi_data.metadata, 'noise_filtered': True}
        )
    
    def _apply_windowing(self, csi_data: CSIData) -> CSIData:
        """对 CSI 数据应用窗函数。"""
        # 应用 Hamming 窗以减少频谱泄漏
        window = scipy.signal.windows.hamming(csi_data.num_subcarriers)
        windowed_amplitude = csi_data.amplitude * window[np.newaxis, :]
        
        return CSIData(
            timestamp=csi_data.timestamp,
            amplitude=windowed_amplitude,
            phase=csi_data.phase,
            frequency=csi_data.frequency,
            bandwidth=csi_data.bandwidth,
            num_subcarriers=csi_data.num_subcarriers,
            num_antennas=csi_data.num_antennas,
            snr=csi_data.snr,
            metadata={**csi_data.metadata, 'windowed': True}
        )
    
    def _normalize_amplitude(self, csi_data: CSIData) -> CSIData:
        """对幅度值进行归一化。"""
        # 归一化到单位方差
        normalized_amplitude = csi_data.amplitude / (np.std(csi_data.amplitude) + 1e-12)
        
        return CSIData(
            timestamp=csi_data.timestamp,
            amplitude=normalized_amplitude,
            phase=csi_data.phase,
            frequency=csi_data.frequency,
            bandwidth=csi_data.bandwidth,
            num_subcarriers=csi_data.num_subcarriers,
            num_antennas=csi_data.num_antennas,
            snr=csi_data.snr,
            metadata={**csi_data.metadata, 'normalized': True}
        )
    
    def _extract_amplitude_features(self, csi_data: CSIData) -> tuple:
        """提取基于幅度的特征。"""
        amplitude_mean = np.mean(csi_data.amplitude, axis=0)
        amplitude_variance = np.var(csi_data.amplitude, axis=0)
        return amplitude_mean, amplitude_variance
    
    def _extract_phase_features(self, csi_data: CSIData) -> np.ndarray:
        """提取基于相位的特征。"""
        # 计算相邻子载波之间的相位差
        phase_diff = np.diff(csi_data.phase, axis=1)
        return np.mean(phase_diff, axis=0)
    
    def _extract_correlation_features(self, csi_data: CSIData) -> np.ndarray:
        """提取天线之间的相关性特征。"""
        # 计算天线之间的相关矩阵
        correlation_matrix = np.corrcoef(csi_data.amplitude)
        return correlation_matrix
    
    def _extract_doppler_features(self, csi_data: CSIData) -> tuple:
        """从时序 CSI 历史中提取多普勒和频域特征。

        使用缓存的平均相位值实现 O(1) 访问，而不是从原始 CSI 帧中重复计算。
        仅使用最近的 `doppler_window` 帧（默认 64 帧），以控制计算时间上界。

        Returns:
            tuple: 以 numpy 数组表示的 `(doppler_shift, power_spectral_density)`
        """
        n_doppler_bins = 64

        if len(self._phase_cache) >= 2:
            # 使用缓存的平均相位值（在 add_to_history 中预先计算）
            # 仅取最近的 doppler_window 帧，以限制计算开销
            window = min(len(self._phase_cache), self._doppler_window)
            start = len(self._phase_cache) - window
            cache_list = list(itertools.islice(self._phase_cache, start, len(self._phase_cache)))
            phase_matrix = np.array(cache_list)

            # 计算相邻帧之间的时序相位差
            phase_diffs = np.diff(phase_matrix, axis=0)

            # 对每个时间步在各子载波上求平均
            mean_phase_diff = np.mean(phase_diffs, axis=1)

            # 对多普勒谱执行 FFT
            doppler_spectrum = np.abs(scipy.fft.fft(mean_phase_diff, n=n_doppler_bins)) ** 2

            # 归一化
            max_val = np.max(doppler_spectrum)
            if max_val > 0:
                doppler_spectrum = doppler_spectrum / max_val

            doppler_shift = doppler_spectrum
        else:
            doppler_shift = np.zeros(n_doppler_bins)

        # 当前帧的功率谱密度
        psd = np.abs(scipy.fft.fft(csi_data.amplitude.flatten(), n=128)) ** 2

        return doppler_shift, psd
    
    def _analyze_motion_patterns(self, features: CSIFeatures) -> float:
        """从特征中分析运动模式。"""
        # 通过方差和相关性模式分析运动情况
        variance_score = np.mean(features.amplitude_variance)
        correlation_score = np.mean(np.abs(features.correlation_matrix - np.eye(features.correlation_matrix.shape[0])))
        
        # 组合各项得分（简化方法）
        motion_score = 0.6 * variance_score + 0.4 * correlation_score
        return np.clip(motion_score, 0.0, 1.0)
    
    def _calculate_detection_confidence(self, features: CSIFeatures, motion_score: float) -> float:
        """根据特征计算检测置信度。"""
        # 组合多个特征指标
        amplitude_indicator = np.mean(features.amplitude_mean) > 0.1
        phase_indicator = np.std(features.phase_difference) > 0.05
        motion_indicator = motion_score > 0.3
        
        # 对各指标加权
        confidence = (0.4 * amplitude_indicator + 0.3 * phase_indicator + 0.3 * motion_indicator)
        return np.clip(confidence, 0.0, 1.0)
    
    def _apply_temporal_smoothing(self, raw_confidence: float) -> float:
        """对检测置信度应用时间平滑。"""
        # 指数移动平均
        smoothed_confidence = (self.smoothing_factor * self.previous_detection_confidence + 
                             (1 - self.smoothing_factor) * raw_confidence)
        
        self.previous_detection_confidence = smoothed_confidence
        return smoothed_confidence
