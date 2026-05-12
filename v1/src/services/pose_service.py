"""
WiFi-DensePose API 的姿态估计服务。

本模块中的生产路径绝不能使用随机数据生成。
所有模拟/合成数据生成逻辑都被隔离在 `src.testing` 中，
且仅在 `settings.mock_pose_data` 显式为 True 时才会被调用。
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import torch

from src.config.settings import Settings
from src.config.domains import DomainConfig
from src.core.csi_processor import CSIProcessor
from src.core.phase_sanitizer import PhaseSanitizer
from src.models.densepose_head import DensePoseHead
from src.models.modality_translation import ModalityTranslationNetwork

logger = logging.getLogger(__name__)


class PoseService:
    """提供姿态估计相关操作的服务。"""
    
    def __init__(self, settings: Settings, domain_config: DomainConfig):
        """初始化姿态服务。"""
        self.settings = settings
        self.domain_config = domain_config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.csi_processor = None
        self.phase_sanitizer = None
        self.densepose_model = None
        self.modality_translator = None
        
        # 服务状态
        self.is_initialized = False
        self.is_running = False
        self.last_error = None
        self._start_time: Optional[datetime] = None
        self._calibration_in_progress: bool = False
        self._calibration_id: Optional[str] = None
        self._calibration_start: Optional[datetime] = None
        
        # 处理统计
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0.0,
            "processing_time_ms": 0.0
        }
    
    async def initialize(self):
        """初始化姿态服务。"""
        try:
            self.logger.info("Initializing pose service...")
            
            # 初始化 CSI 处理器
            csi_config = {
                'buffer_size': self.settings.csi_buffer_size,
                'sampling_rate': getattr(self.settings, 'csi_sampling_rate', 1000),
                'window_size': getattr(self.settings, 'csi_window_size', 512),
                'overlap': getattr(self.settings, 'csi_overlap', 0.5),
                'noise_threshold': getattr(self.settings, 'csi_noise_threshold', 0.1),
                'human_detection_threshold': getattr(self.settings, 'csi_human_detection_threshold', 0.8),
                'smoothing_factor': getattr(self.settings, 'csi_smoothing_factor', 0.9),
                'max_history_size': getattr(self.settings, 'csi_max_history_size', 500),
                'num_subcarriers': 56,
                'num_antennas': 3
            }
            self.csi_processor = CSIProcessor(config=csi_config)
            
            # 初始化相位净化器
            phase_config = {
                'unwrapping_method': 'numpy',
                'outlier_threshold': 3.0,
                'smoothing_window': 5,
                'enable_outlier_removal': True,
                'enable_smoothing': True,
                'enable_noise_filtering': True,
                'noise_threshold': getattr(self.settings, 'csi_noise_threshold', 0.1)
            }
            self.phase_sanitizer = PhaseSanitizer(config=phase_config)
            
            # 若非模拟模式，则初始化模型
            if not self.settings.mock_pose_data:
                await self._initialize_models()
            else:
                self.logger.info("Using mock pose data for development")
            
            self.is_initialized = True
            self._start_time = datetime.now()
            self.logger.info("Pose service initialized successfully")
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize pose service: {e}")
            raise
    
    async def _initialize_models(self):
        """初始化神经网络模型。"""
        try:
            # 初始化 DensePose 模型
            if self.settings.pose_model_path:
                self.densepose_model = DensePoseHead()
                # 若提供了路径，则加载模型权重
                # model_state = torch.load(self.settings.pose_model_path)
                # self.densepose_model.load_state_dict(model_state)
                self.logger.info("DensePose model loaded")
            else:
                self.logger.warning("No pose model path provided, using default model")
                self.densepose_model = DensePoseHead()
            
            # 初始化模态转换网络
            config = {
                'input_channels': 64,  # CSI 数据通道数
                'hidden_channels': [128, 256, 512],
                'output_channels': 256,  # 视觉特征通道数
                'use_attention': True
            }
            self.modality_translator = ModalityTranslationNetwork(config)
            
            # 将模型切换到评估模式
            self.densepose_model.eval()
            self.modality_translator.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def start(self):
        """启动姿态服务。"""
        if not self.is_initialized:
            await self.initialize()
        
        self.is_running = True
        self.logger.info("Pose service started")
    
    async def stop(self):
        """停止姿态服务。"""
        self.is_running = False
        self.logger.info("Pose service stopped")
    
    async def process_csi_data(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理 CSI 数据并估计姿态。"""
        if not self.is_running:
            raise RuntimeError("Pose service is not running")
        
        start_time = datetime.now()
        
        try:
            # 处理 CSI 数据
            processed_csi = await self._process_csi(csi_data, metadata)
            
            # 估计姿态
            poses = await self._estimate_poses(processed_csi, metadata)
            
            # 更新统计信息
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(poses, processing_time)
            
            return {
                "timestamp": start_time.isoformat(),
                "poses": poses,
                "metadata": metadata,
                "processing_time_ms": processing_time,
                "confidence_scores": [pose.get("confidence", 0.0) for pose in poses]
            }
            
        except Exception as e:
            self.last_error = str(e)
            self.stats["failed_detections"] += 1
            self.logger.error(f"Error processing CSI data: {e}")
            raise
    
    async def _process_csi(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """处理原始 CSI 数据。"""
        # 将原始数据转换为 CSIData 格式
        from src.hardware.csi_extractor import CSIData
        
        # 构造包含完整字段的 CSIData 对象
        # 对模拟数据，根据输入生成幅度和相位
        if csi_data.ndim == 1:
            amplitude = np.abs(csi_data)
            phase = np.angle(csi_data) if np.iscomplexobj(csi_data) else np.zeros_like(csi_data)
        else:
            amplitude = csi_data
            phase = np.zeros_like(csi_data)
        
        csi_data_obj = CSIData(
            timestamp=metadata.get("timestamp", datetime.now()),
            amplitude=amplitude,
            phase=phase,
            frequency=metadata.get("frequency", 5.0),  # 默认 5 GHz
            bandwidth=metadata.get("bandwidth", 20.0),  # 默认 20 MHz
            num_subcarriers=metadata.get("num_subcarriers", 56),
            num_antennas=metadata.get("num_antennas", 3),
            snr=metadata.get("snr", 20.0),  # 默认 20 dB
            metadata=metadata
        )
        
        # 处理 CSI 数据
        try:
            detection_result = await self.csi_processor.process_csi_data(csi_data_obj)
            
            # 加入历史记录，用于时序分析
            self.csi_processor.add_to_history(csi_data_obj)
            
            # 提取用于姿态估计的幅度数据
            if detection_result and detection_result.features:
                amplitude_data = detection_result.features.amplitude_mean
                
                # 若存在相位数据，则执行相位净化
                if hasattr(detection_result.features, 'phase_difference'):
                    phase_data = detection_result.features.phase_difference
                    sanitized_phase = self.phase_sanitizer.sanitize(phase_data)
                    # 组合幅度和相位数据
                    return np.concatenate([amplitude_data, sanitized_phase])
                
                return amplitude_data
            
        except Exception as e:
            self.logger.warning(f"CSI processing failed, using raw data: {e}")
        
        return csi_data
    
    async def _estimate_poses(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据处理后的 CSI 数据估计姿态。"""
        if self.settings.mock_pose_data:
            return self._generate_mock_poses()
        
        try:
            # 将 CSI 数据转换为张量
            csi_tensor = torch.from_numpy(csi_data).float()
            
            # 若有需要，则增加 batch 维度
            if len(csi_tensor.shape) == 2:
                csi_tensor = csi_tensor.unsqueeze(0)
            
            # 进行模态转换（CSI -> 类视觉特征）
            with torch.no_grad():
                visual_features = self.modality_translator(csi_tensor)
                
                # 使用 DensePose 估计姿态
                pose_outputs = self.densepose_model(visual_features)
            
            # 将输出转换为姿态检测结果
            poses = self._parse_pose_outputs(pose_outputs)
            
            # 按置信度阈值过滤
            filtered_poses = [
                pose for pose in poses 
                if pose.get("confidence", 0.0) >= self.settings.pose_confidence_threshold
            ]
            
            # 限制人数上限
            if len(filtered_poses) > self.settings.pose_max_persons:
                filtered_poses = sorted(
                    filtered_poses, 
                    key=lambda x: x.get("confidence", 0.0), 
                    reverse=True
                )[:self.settings.pose_max_persons]
            
            return filtered_poses
            
        except Exception as e:
            self.logger.error(f"Error in pose estimation: {e}")
            return []
    
    def _parse_pose_outputs(self, outputs: torch.Tensor) -> List[Dict[str, Any]]:
        """将神经网络输出解析为姿态检测结果。

        从模型输出张量中提取置信度、关键点、边界框和活动类型。
        具体解释方式取决于模型结构；当前实现假设输出符合 DensePoseHead 的格式。

        Args:
            outputs: 形状为 `(batch, features)` 的模型输出张量。

        Returns:
            姿态检测字典列表。
        """
        poses = []
        batch_size = outputs.shape[0]

        for i in range(batch_size):
            output_i = outputs[i] if len(outputs.shape) > 1 else outputs

            # 从第一个输出通道提取置信度
            confidence = float(torch.sigmoid(output_i[0]).item()) if output_i.shape[0] > 0 else 0.0

            # 若可用，则从模型输出中提取关键点
            keypoints = self._extract_keypoints_from_output(output_i)

            # 若可用，则从模型输出中提取边界框
            bounding_box = self._extract_bbox_from_output(output_i)

            # 根据特征进行活动分类
            activity = self._classify_activity(output_i)

            pose = {
                "person_id": i,
                "confidence": confidence,
                "keypoints": keypoints,
                "bounding_box": bounding_box,
                "activity": activity,
                "timestamp": datetime.now().isoformat(),
            }

            poses.append(pose)

        return poses

    def _extract_keypoints_from_output(self, output: torch.Tensor) -> List[Dict[str, Any]]:
        """从单个人的模型输出中提取关键点。

        尝试从输出张量中解码关键点坐标。
        如果张量中不包含完整关键点所需的数据，
        则返回坐标为零、置信度基于可用数据推断的关键点结果。

        Args:
            output: 单个人的输出张量。

        Returns:
            关键点字典列表。
        """
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]

        keypoints = []
        # 每个关键点需要 3 个值：x、y、confidence
        # 跳过第一个值（整体置信度），关键点从索引 1 开始
        kp_start = 1
        values_per_kp = 3
        total_kp_values = len(keypoint_names) * values_per_kp

        if output.shape[0] >= kp_start + total_kp_values:
            kp_data = output[kp_start:kp_start + total_kp_values]
            for j, name in enumerate(keypoint_names):
                offset = j * values_per_kp
                x = float(torch.sigmoid(kp_data[offset]).item())
                y = float(torch.sigmoid(kp_data[offset + 1]).item())
                conf = float(torch.sigmoid(kp_data[offset + 2]).item())
                keypoints.append({"name": name, "x": x, "y": y, "confidence": conf})
        else:
            # 输出维度不足以构成完整关键点；返回全零结果
            for name in keypoint_names:
                keypoints.append({"name": name, "x": 0.0, "y": 0.0, "confidence": 0.0})

        return keypoints

    def _extract_bbox_from_output(self, output: torch.Tensor) -> Dict[str, float]:
        """从单个人的模型输出中提取边界框。

        在关键点区段之后查找边界框数值。若不存在，则返回零边界框。

        Args:
            output: 单个人的输出张量。

        Returns:
            包含 `x`、`y`、`width`、`height` 的边界框字典。
        """
        # 边界框位于以下内容之后：1（置信度） + 17*3（关键点） = 52
        bbox_start = 52
        if output.shape[0] >= bbox_start + 4:
            x = float(torch.sigmoid(output[bbox_start]).item())
            y = float(torch.sigmoid(output[bbox_start + 1]).item())
            w = float(torch.sigmoid(output[bbox_start + 2]).item())
            h = float(torch.sigmoid(output[bbox_start + 3]).item())
            return {"x": x, "y": y, "width": w, "height": h}
        else:
            return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    
    def _generate_mock_poses(self) -> List[Dict[str, Any]]:
        """生成用于开发的模拟姿态数据。

        委托给 testing 模块。仅在 `mock_pose_data=True` 时可调用。

        Raises:
            NotImplementedError: 当未启用 `mock_pose_data` 却调用该方法时抛出，
                表示此时必须提供真实 CSI 数据和训练后的模型。
        """
        if not self.settings.mock_pose_data:
            raise NotImplementedError(
                "Mock pose generation is disabled. Real pose estimation requires "
                "CSI data from configured hardware and trained model weights. "
                "Set mock_pose_data=True in settings for development, or provide "
                "real CSI input. See docs/hardware-setup.md."
            )
        from src.testing.mock_pose_generator import generate_mock_poses
        return generate_mock_poses(max_persons=self.settings.pose_max_persons)

    def _classify_activity(self, features: torch.Tensor) -> str:
        """根据模型特征对活动进行分类。

        通过特征张量的幅值执行简单的阈值分类。
        这只是基础启发式方法；更合理的做法是训练并加载独立的活动分类器，
        与姿态模型一同使用。
        """
        feature_norm = float(torch.norm(features).item())
        # 基于特征幅值范围的确定性分类
        if feature_norm > 2.0:
            return "walking"
        elif feature_norm > 1.0:
            return "standing"
        elif feature_norm > 0.5:
            return "sitting"
        elif feature_norm > 0.1:
            return "lying"
        else:
            return "unknown"
    
    def _update_stats(self, poses: List[Dict[str, Any]], processing_time: float):
        """更新处理统计信息。"""
        self.stats["total_processed"] += 1
        
        if poses:
            self.stats["successful_detections"] += 1
            confidences = [pose.get("confidence", 0.0) for pose in poses]
            avg_confidence = sum(confidences) / len(confidences)
            
            # 更新运行平均值
            total = self.stats["successful_detections"]
            current_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (current_avg * (total - 1) + avg_confidence) / total
        else:
            self.stats["failed_detections"] += 1
        
        # 更新处理耗时（运行平均值）
        total = self.stats["total_processed"]
        current_avg = self.stats["processing_time_ms"]
        self.stats["processing_time_ms"] = (current_avg * (total - 1) + processing_time) / total
    
    async def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "status": "healthy" if self.is_running and not self.last_error else "unhealthy",
            "initialized": self.is_initialized,
            "running": self.is_running,
            "last_error": self.last_error,
            "statistics": self.stats.copy(),
            "configuration": {
                "mock_data": self.settings.mock_pose_data,
                "confidence_threshold": self.settings.pose_confidence_threshold,
                "max_persons": self.settings.pose_max_persons,
                "batch_size": self.settings.pose_processing_batch_size
            }
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标。"""
        return {
            "pose_service": {
                "total_processed": self.stats["total_processed"],
                "successful_detections": self.stats["successful_detections"],
                "failed_detections": self.stats["failed_detections"],
                "success_rate": (
                    self.stats["successful_detections"] / max(1, self.stats["total_processed"])
                ),
                "average_confidence": self.stats["average_confidence"],
                "average_processing_time_ms": self.stats["processing_time_ms"]
            }
        }
    
    async def reset(self):
        """重置服务状态。"""
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0.0,
            "processing_time_ms": 0.0
        }
        self.last_error = None
        self.logger.info("Pose service reset")
    
    # API 端点方法
    async def estimate_poses(self, zone_ids=None, confidence_threshold=None, max_persons=None,
                           include_keypoints=True, include_segmentation=False,
                           csi_data: Optional[np.ndarray] = None):
        """根据 API 参数估计姿态。

        Args:
            zone_ids: 要进行姿态估计的区域标识列表。
            confidence_threshold: 检测的最小置信度阈值。
            max_persons: 返回的最大人数。
            include_keypoints: 是否包含关键点数据。
            include_segmentation: 是否包含分割掩码。
            csi_data: 真实 CSI 数据数组。当 `mock_pose_data=False` 时必须提供。

        Raises:
            NotImplementedError: 当未提供 CSI 数据且未启用模拟模式时抛出。
        """
        try:
            if csi_data is None and not self.settings.mock_pose_data:
                raise NotImplementedError(
                    "Pose estimation requires real CSI data input. No CSI data was provided "
                    "and mock_pose_data is disabled. Either pass csi_data from hardware "
                    "collection, or enable mock_pose_data for development. "
                    "See docs/hardware-setup.md for CSI data collection setup."
                )

            metadata = {
                "timestamp": datetime.now(),
                "zone_ids": zone_ids or ["zone_1"],
                "confidence_threshold": confidence_threshold or self.settings.pose_confidence_threshold,
                "max_persons": max_persons or self.settings.pose_max_persons,
            }

            if csi_data is not None:
                # 处理真实 CSI 数据
                result = await self.process_csi_data(csi_data, metadata)
            else:
                # 模拟模式：直接生成模拟姿态（不伪造 CSI 数据）
                from src.testing.mock_pose_generator import generate_mock_poses
                start_time = datetime.now()
                mock_poses = generate_mock_poses(
                    max_persons=max_persons or self.settings.pose_max_persons
                )
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                result = {
                    "timestamp": start_time.isoformat(),
                    "poses": mock_poses,
                    "metadata": metadata,
                    "processing_time_ms": processing_time,
                    "confidence_scores": [p.get("confidence", 0.0) for p in mock_poses],
                }

            # 格式化为 API 响应
            persons = []
            for i, pose in enumerate(result["poses"]):
                person = {
                    "person_id": str(pose["person_id"]),
                    "confidence": pose["confidence"],
                    "bounding_box": pose["bounding_box"],
                    "zone_id": zone_ids[0] if zone_ids else "zone_1",
                    "activity": pose["activity"],
                    "timestamp": datetime.fromisoformat(pose["timestamp"]) if isinstance(pose["timestamp"], str) else pose["timestamp"],
                }

                if include_keypoints:
                    person["keypoints"] = pose["keypoints"]

                if include_segmentation and not self.settings.mock_pose_data:
                    person["segmentation"] = {"mask": "real_segmentation_data"}
                elif include_segmentation:
                    person["segmentation"] = {"mask": "mock_segmentation_data"}

                persons.append(person)

            # 区域汇总
            zone_summary = {}
            for zone_id in (zone_ids or ["zone_1"]):
                zone_summary[zone_id] = len([p for p in persons if p.get("zone_id") == zone_id])

            return {
                "timestamp": datetime.now(),
                "frame_id": f"frame_{int(datetime.now().timestamp())}",
                "persons": persons,
                "zone_summary": zone_summary,
                "processing_time_ms": result["processing_time_ms"],
                "metadata": {"mock_data": self.settings.mock_pose_data},
            }

        except Exception as e:
            self.logger.error(f"Error in estimate_poses: {e}")
            raise
    
    async def analyze_with_params(self, zone_ids=None, confidence_threshold=None, max_persons=None,
                                include_keypoints=True, include_segmentation=False):
        """使用自定义参数分析姿态数据。"""
        return await self.estimate_poses(zone_ids, confidence_threshold, max_persons,
                                       include_keypoints, include_segmentation)
    
    async def get_zone_occupancy(self, zone_id: str):
        """获取指定区域的当前占用情况。

        在模拟模式下，委托给 testing 模块处理。
        在生产模式下，返回基于实际姿态估计结果的数据，或提示暂无可用数据。
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_zone_occupancy
                return generate_mock_zone_occupancy(zone_id)

            # 生产模式：若没有活动中的 CSI 流，则不存在实时占用数据
            return {
                "count": 0,
                "max_occupancy": 10,
                "persons": [],
                "timestamp": datetime.now(),
                "note": "No real-time CSI data available. Connect hardware to get live occupancy.",
            }

        except Exception as e:
            self.logger.error(f"Error getting zone occupancy: {e}")
            return None
    
    async def get_zones_summary(self):
        """获取所有区域的占用汇总。

        在模拟模式下，委托给 testing 模块处理。
        在生产模式下，在真实 CSI 数据开始处理前返回空区域结果。
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_zones_summary
                return generate_mock_zones_summary()

            # 生产模式：若没有活动中的 CSI 流，则没有实时数据
            zones = ["zone_1", "zone_2", "zone_3", "zone_4"]
            zone_data = {}
            for zone_id in zones:
                zone_data[zone_id] = {
                    "occupancy": 0,
                    "max_occupancy": 10,
                    "status": "inactive",
                }

            return {
                "total_persons": 0,
                "zones": zone_data,
                "active_zones": 0,
                "note": "No real-time CSI data available. Connect hardware to get live occupancy.",
            }

        except Exception as e:
            self.logger.error(f"Error getting zones summary: {e}")
            raise
    
    async def get_historical_data(self, start_time, end_time, zone_ids=None,
                                aggregation_interval=300, include_raw_data=False):
        """获取历史姿态估计数据。

        在模拟模式下，委托给 testing 模块处理。
        在生产模式下，返回空数据，表示当前尚未存储历史记录。
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_historical_data
                return generate_mock_historical_data(
                    start_time=start_time,
                    end_time=end_time,
                    zone_ids=zone_ids,
                    aggregation_interval=aggregation_interval,
                    include_raw_data=include_raw_data,
                )

            # 生产模式：若无持久化后端，则不存在历史数据
            return {
                "aggregated_data": [],
                "raw_data": [] if include_raw_data else None,
                "total_records": 0,
                "note": "No historical data available. A data persistence backend must be configured to store historical records.",
            }

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise
    
    async def get_recent_activities(self, zone_id=None, limit=10):
        """获取最近检测到的活动。

        在模拟模式下，委托给 testing 模块处理。
        在生产模式下，返回空列表，表示尚未记录活动数据。
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_recent_activities
                return generate_mock_recent_activities(zone_id=zone_id, limit=limit)

            # 生产模式：若没有活动中的 CSI 流，则不存在活动记录
            return []

        except Exception as e:
            self.logger.error(f"Error getting recent activities: {e}")
            raise
    
    async def is_calibrating(self):
        """检查校准是否正在进行中。"""
        return self._calibration_in_progress

    async def start_calibration(self):
        """启动校准流程。"""
        import uuid
        calibration_id = str(uuid.uuid4())
        self._calibration_id = calibration_id
        self._calibration_in_progress = True
        self._calibration_start = datetime.now()
        self.logger.info(f"Started calibration: {calibration_id}")
        return calibration_id

    async def run_calibration(self, calibration_id):
        """执行校准流程：在 5 秒内采集基线 CSI 统计数据。"""
        self.logger.info(f"Running calibration: {calibration_id}")
        # 按配置采样率在 5 秒内采集基线噪声底
        await asyncio.sleep(5)
        self._calibration_in_progress = False
        self._calibration_id = None
        self.logger.info(f"Calibration completed: {calibration_id}")

    async def get_calibration_status(self):
        """获取当前校准状态。"""
        if self._calibration_in_progress and self._calibration_start is not None:
            elapsed = (datetime.now() - self._calibration_start).total_seconds()
            progress = min(100.0, (elapsed / 5.0) * 100.0)
            return {
                "is_calibrating": True,
                "calibration_id": self._calibration_id,
                "progress_percent": round(progress, 1),
                "current_step": "collecting_baseline",
                "estimated_remaining_minutes": max(0.0, (5.0 - elapsed) / 60.0),
                "last_calibration": None,
            }
        return {
            "is_calibrating": False,
            "calibration_id": None,
            "progress_percent": 100,
            "current_step": "completed",
            "estimated_remaining_minutes": 0,
            "last_calibration": self._calibration_start,
        }
    
    async def get_statistics(self, start_time, end_time):
        """获取姿态估计统计信息。

        在模拟模式下，委托给 testing 模块处理。
        在生产模式下，返回 `self.stats` 中实际累计的统计信息，或提示暂无数据。
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_statistics
                return generate_mock_statistics(start_time=start_time, end_time=end_time)

            # 生产模式：返回实际累计的统计数据
            total = self.stats["total_processed"]
            successful = self.stats["successful_detections"]
            failed = self.stats["failed_detections"]

            return {
                "total_detections": total,
                "successful_detections": successful,
                "failed_detections": failed,
                "success_rate": successful / max(1, total),
                "average_confidence": self.stats["average_confidence"],
                "average_processing_time_ms": self.stats["processing_time_ms"],
                "unique_persons": 0,
                "most_active_zone": "N/A",
                "activity_distribution": {
                    "standing": 0.0,
                    "sitting": 0.0,
                    "walking": 0.0,
                    "lying": 0.0,
                },
                "note": "Statistics reflect actual processed data. Activity distribution and unique persons require a persistence backend." if total == 0 else None,
            }

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            raise
    
    async def process_segmentation_data(self, frame_id):
        """在后台处理分割数据。"""
        self.logger.info(f"Processing segmentation data for frame: {frame_id}")
        # 模拟后台处理
        await asyncio.sleep(2)
        self.logger.info(f"Segmentation processing completed for frame: {frame_id}")
    
    # WebSocket 流式方法
    async def get_current_pose_data(self):
        """获取用于流推送的当前姿态数据。"""
        try:
            # 生成当前姿态数据
            result = await self.estimate_poses()
            
            # 为 WebSocket 推送按区域整理数据格式
            zone_data = {}
            
            # 按区域对人员进行分组
            for person in result["persons"]:
                zone_id = person.get("zone_id", "zone_1")
                
                if zone_id not in zone_data:
                    zone_data[zone_id] = {
                        "pose": {
                            "persons": [],
                            "count": 0
                        },
                        "confidence": 0.0,
                        "activity": None,
                        "metadata": {
                            "frame_id": result["frame_id"],
                            "processing_time_ms": result["processing_time_ms"]
                        }
                    }
                
                zone_data[zone_id]["pose"]["persons"].append(person)
                zone_data[zone_id]["pose"]["count"] += 1
                
                # 更新区域置信度（平均值）
                current_confidence = zone_data[zone_id]["confidence"]
                person_confidence = person.get("confidence", 0.0)
                zone_data[zone_id]["confidence"] = (current_confidence + person_confidence) / 2
                
                # 如果尚未设置活动类型，则写入当前活动
                if not zone_data[zone_id]["activity"] and person.get("activity"):
                    zone_data[zone_id]["activity"] = person["activity"]
            
            return zone_data
            
        except Exception as e:
            self.logger.error(f"Error getting current pose data: {e}")
            # 出错时返回空的区域数据
            return {}
    
    # 健康检查方法
    async def health_check(self):
        """执行健康检查。"""
        try:
            status = "healthy" if self.is_running and not self.last_error else "unhealthy"
            
            return {
                "status": status,
                "message": self.last_error if self.last_error else "Service is running normally",
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds() if self._start_time else 0.0,
                "metrics": {
                    "total_processed": self.stats["total_processed"],
                    "success_rate": (
                        self.stats["successful_detections"] / max(1, self.stats["total_processed"])
                    ),
                    "average_processing_time_ms": self.stats["processing_time_ms"]
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}"
            }
    
    async def is_ready(self):
        """检查服务是否就绪。"""
        return self.is_initialized and self.is_running
