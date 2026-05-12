"""
WiFi-DensePose API 的指标采集服务
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """单个指标数据点。"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """指标数据点的时间序列。"""
    name: str
    description: str
    unit: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None):
        """添加一个指标数据点。"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self.points.append(point)
    
    def get_latest(self) -> Optional[MetricPoint]:
        """获取最新的指标数据点。"""
        return self.points[-1] if self.points else None
    
    def get_average(self, duration: timedelta) -> Optional[float]:
        """获取指定时间范围内的平均值。"""
        cutoff = datetime.utcnow() - duration
        relevant_points = [
            point for point in self.points
            if point.timestamp >= cutoff
        ]
        
        if not relevant_points:
            return None
        
        return sum(point.value for point in relevant_points) / len(relevant_points)
    
    def get_max(self, duration: timedelta) -> Optional[float]:
        """获取指定时间范围内的最大值。"""
        cutoff = datetime.utcnow() - duration
        relevant_points = [
            point for point in self.points
            if point.timestamp >= cutoff
        ]
        
        if not relevant_points:
            return None
        
        return max(point.value for point in relevant_points)


class MetricsService:
    """用于采集和管理应用指标的服务。"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._metrics: Dict[str, MetricSeries] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._start_time = time.time()
        self._initialized = False
        self._running = False
        
        # 初始化标准指标
        self._initialize_standard_metrics()
    
    def _initialize_standard_metrics(self):
        """初始化标准系统指标和应用指标。"""
        self._metrics.update({
            # 系统指标
            "system_cpu_usage": MetricSeries(
                "system_cpu_usage", "System CPU usage percentage", "percent"
            ),
            "system_memory_usage": MetricSeries(
                "system_memory_usage", "System memory usage percentage", "percent"
            ),
            "system_disk_usage": MetricSeries(
                "system_disk_usage", "System disk usage percentage", "percent"
            ),
            "system_network_bytes_sent": MetricSeries(
                "system_network_bytes_sent", "Network bytes sent", "bytes"
            ),
            "system_network_bytes_recv": MetricSeries(
                "system_network_bytes_recv", "Network bytes received", "bytes"
            ),
            
            # 应用指标
            "app_requests_total": MetricSeries(
                "app_requests_total", "Total HTTP requests", "count"
            ),
            "app_request_duration": MetricSeries(
                "app_request_duration", "HTTP request duration", "seconds"
            ),
            "app_active_connections": MetricSeries(
                "app_active_connections", "Active WebSocket connections", "count"
            ),
            "app_pose_detections": MetricSeries(
                "app_pose_detections", "Pose detections performed", "count"
            ),
            "app_pose_processing_time": MetricSeries(
                "app_pose_processing_time", "Pose processing time", "seconds"
            ),
            "app_csi_data_points": MetricSeries(
                "app_csi_data_points", "CSI data points processed", "count"
            ),
            "app_stream_fps": MetricSeries(
                "app_stream_fps", "Streaming frames per second", "fps"
            ),
            
            # 错误指标
            "app_errors_total": MetricSeries(
                "app_errors_total", "Total application errors", "count"
            ),
            "app_http_errors": MetricSeries(
                "app_http_errors", "HTTP errors", "count"
            ),
        })
    
    async def initialize(self):
        """初始化指标服务。"""
        if self._initialized:
            return
        
        logger.info("Initializing metrics service")
        self._initialized = True
        logger.info("Metrics service initialized")
    
    async def start(self):
        """启动指标服务。"""
        if not self._initialized:
            await self.initialize()
        
        self._running = True
        logger.info("Metrics service started")
    
    async def shutdown(self):
        """关闭指标服务。"""
        self._running = False
        logger.info("Metrics service shut down")
    
    async def collect_metrics(self):
        """采集所有指标。"""
        if not self._running:
            return
        
        logger.debug("Collecting metrics")
        
        # 采集系统指标
        await self._collect_system_metrics()
        
        # 采集应用指标
        await self._collect_application_metrics()
        
        logger.debug("Metrics collection completed")
    
    async def _collect_system_metrics(self):
        """采集系统级指标。"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self._metrics["system_cpu_usage"].add_point(cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self._metrics["system_memory_usage"].add_point(memory.percent)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._metrics["system_disk_usage"].add_point(disk_percent)
            
            # 网络 I/O
            network = psutil.net_io_counters()
            self._metrics["system_network_bytes_sent"].add_point(network.bytes_sent)
            self._metrics["system_network_bytes_recv"].add_point(network.bytes_recv)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """采集应用特定指标。"""
        try:
            # 在此处导入以避免循环依赖
            from src.api.websocket.connection_manager import connection_manager
            
            # 活跃连接数
            connection_stats = await connection_manager.get_connection_stats()
            active_connections = connection_stats.get("active_connections", 0)
            self._metrics["app_active_connections"].add_point(active_connections)
            
            # 将计数器作为指标写入
            for name, value in self._counters.items():
                if name in self._metrics:
                    self._metrics[name].add_point(value)
            
            # 将仪表值作为指标写入
            for name, value in self._gauges.items():
                if name in self._metrics:
                    self._metrics[name].add_point(value)
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """递增计数器指标。"""
        self._counters[name] += value
        
        if name in self._metrics:
            self._metrics[name].add_point(self._counters[name], labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """设置仪表指标值。"""
        self._gauges[name] = value
        
        if name in self._metrics:
            self._metrics[name].add_point(value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录直方图指标值。"""
        self._histograms[name].append(value)
        
        # 仅保留最近 1000 个值
        if len(self._histograms[name]) > 1000:
            self._histograms[name] = self._histograms[name][-1000:]
        
        if name in self._metrics:
            self._metrics[name].add_point(value, labels)
    
    def time_function(self, metric_name: str):
        """用于统计函数执行耗时的装饰器。"""
        def decorator(func):
            import functools
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.record_histogram(metric_name, duration)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.record_histogram(metric_name, duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """按名称获取指标序列。"""
        return self._metrics.get(name)
    
    def get_metric_value(self, name: str) -> Optional[float]:
        """获取指标的最新值。"""
        metric = self._metrics.get(name)
        if metric:
            latest = metric.get_latest()
            return latest.value if latest else None
        return None
    
    def get_counter_value(self, name: str) -> float:
        """获取当前计数器值。"""
        return self._counters.get(name, 0.0)
    
    def get_gauge_value(self, name: str) -> Optional[float]:
        """获取当前仪表值。"""
        return self._gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """获取直方图统计信息。"""
        values = self._histograms.get(name, [])
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "sum": sum(sorted_values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p90": sorted_values[int(count * 0.9)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)],
        }
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """获取当前所有指标。"""
        metrics = {}
        
        # 当前指标值
        for name, metric_series in self._metrics.items():
            latest = metric_series.get_latest()
            if latest:
                metrics[name] = {
                    "value": latest.value,
                    "timestamp": latest.timestamp.isoformat(),
                    "description": metric_series.description,
                    "unit": metric_series.unit,
                    "labels": latest.labels
                }
        
        # 计数器值
        metrics.update({
            f"counter_{name}": value
            for name, value in self._counters.items()
        })
        
        # 仪表值
        metrics.update({
            f"gauge_{name}": value
            for name, value in self._gauges.items()
        })
        
        # 直方图统计信息
        for name, values in self._histograms.items():
            if values:
                stats = self.get_histogram_stats(name)
                metrics[f"histogram_{name}"] = stats
        
        return metrics
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标摘要。"""
        return {
            "cpu_usage": self.get_metric_value("system_cpu_usage"),
            "memory_usage": self.get_metric_value("system_memory_usage"),
            "disk_usage": self.get_metric_value("system_disk_usage"),
            "network_bytes_sent": self.get_metric_value("system_network_bytes_sent"),
            "network_bytes_recv": self.get_metric_value("system_network_bytes_recv"),
        }
    
    async def get_application_metrics(self) -> Dict[str, Any]:
        """获取应用指标摘要。"""
        return {
            "requests_total": self.get_counter_value("app_requests_total"),
            "active_connections": self.get_metric_value("app_active_connections"),
            "pose_detections": self.get_counter_value("app_pose_detections"),
            "csi_data_points": self.get_counter_value("app_csi_data_points"),
            "errors_total": self.get_counter_value("app_errors_total"),
            "uptime_seconds": time.time() - self._start_time,
            "request_duration_stats": self.get_histogram_stats("app_request_duration"),
            "pose_processing_time_stats": self.get_histogram_stats("app_pose_processing_time"),
        }
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能指标摘要。"""
        one_hour = timedelta(hours=1)
        
        return {
            "system": {
                "cpu_avg_1h": self._metrics["system_cpu_usage"].get_average(one_hour),
                "memory_avg_1h": self._metrics["system_memory_usage"].get_average(one_hour),
                "cpu_max_1h": self._metrics["system_cpu_usage"].get_max(one_hour),
                "memory_max_1h": self._metrics["system_memory_usage"].get_max(one_hour),
            },
            "application": {
                "avg_request_duration": self.get_histogram_stats("app_request_duration").get("mean"),
                "avg_pose_processing_time": self.get_histogram_stats("app_pose_processing_time").get("mean"),
                "total_requests": self.get_counter_value("app_requests_total"),
                "total_errors": self.get_counter_value("app_errors_total"),
                "error_rate": (
                    self.get_counter_value("app_errors_total") / 
                    max(self.get_counter_value("app_requests_total"), 1)
                ) * 100,
            }
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """获取指标服务状态。"""
        return {
            "status": "healthy" if self._running else "stopped",
            "initialized": self._initialized,
            "running": self._running,
            "metrics_count": len(self._metrics),
            "counters_count": len(self._counters),
            "gauges_count": len(self._gauges),
            "histograms_count": len(self._histograms),
            "uptime": time.time() - self._start_time
        }
    
    def reset_metrics(self):
        """重置所有指标。"""
        logger.info("Resetting all metrics")
        
        # 清空指标数据点，但保留序列定义
        for metric_series in self._metrics.values():
            metric_series.points.clear()
        
        # 重置计数器、仪表和直方图
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        
        logger.info("All metrics reset")
