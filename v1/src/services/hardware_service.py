"""
WiFi-DensePose API 的硬件接口服务
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import numpy as np

from src.config.settings import Settings
from src.config.domains import DomainConfig
from src.core.router_interface import RouterInterface

logger = logging.getLogger(__name__)


class HardwareService:
    """提供硬件接口相关操作的服务。"""
    
    def __init__(self, settings: Settings, domain_config: DomainConfig):
        """初始化硬件服务。"""
        self.settings = settings
        self.domain_config = domain_config
        self.logger = logging.getLogger(__name__)
        
        # 路由器接口
        self.router_interfaces: Dict[str, RouterInterface] = {}
        
        # 服务状态
        self.is_running = False
        self.last_error = None
        
        # 数据采集统计
        self.stats = {
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "average_sample_rate": 0.0,
            "last_sample_time": None,
            "connected_routers": 0
        }
        
        # 后台任务
        self.collection_task = None
        self.monitoring_task = None
        
        # 数据缓冲区
        self.recent_samples = []
        self.max_recent_samples = 1000
    
    async def initialize(self):
        """初始化硬件服务。"""
        await self.start()
    
    async def start(self):
        """启动硬件服务。"""
        if self.is_running:
            return
        
        try:
            self.logger.info("Starting hardware service...")
            
            # 初始化路由器接口
            await self._initialize_routers()
            
            self.is_running = True
            
            # 启动后台任务
            if not self.settings.mock_hardware:
                self.collection_task = asyncio.create_task(self._data_collection_loop())
            
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("Hardware service started successfully")
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to start hardware service: {e}")
            raise
    
    async def stop(self):
        """停止硬件服务。"""
        self.is_running = False
        
        # 取消后台任务
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 断开与各路由器的连接
        await self._disconnect_routers()
        
        self.logger.info("Hardware service stopped")
    
    async def _initialize_routers(self):
        """初始化路由器接口。"""
        try:
            # 从领域配置中获取路由器配置
            routers = self.domain_config.get_all_routers()
            
            for router_config in routers:
                if not router_config.enabled:
                    continue
                
                router_id = router_config.router_id
                
                # 创建路由器接口
                router_interface = RouterInterface(
                    router_id=router_id,
                    host=router_config.ip_address,
                    port=getattr(router_config, 'ssh_port', 22),
                    username=getattr(router_config, 'ssh_username', None) or self.settings.router_ssh_username,
                    password=getattr(router_config, 'ssh_password', None) or self.settings.router_ssh_password,
                    interface=router_config.interface,
                    mock_mode=self.settings.mock_hardware
                )
                
                # 连接路由器（即使在模拟模式下也执行连接）
                await router_interface.connect()
                
                self.router_interfaces[router_id] = router_interface
                self.logger.info(f"Router interface initialized: {router_id}")
            
            self.stats["connected_routers"] = len(self.router_interfaces)
            
            if not self.router_interfaces:
                self.logger.warning("No router interfaces configured")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize routers: {e}")
            raise
    
    async def _disconnect_routers(self):
        """断开所有路由器连接。"""
        for router_id, interface in self.router_interfaces.items():
            try:
                await interface.disconnect()
                self.logger.info(f"Disconnected from router: {router_id}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from router {router_id}: {e}")
        
        self.router_interfaces.clear()
        self.stats["connected_routers"] = 0
    
    async def _data_collection_loop(self):
        """用于数据采集的后台循环。"""
        try:
            while self.is_running:
                start_time = time.time()
                
                # 从所有路由器采集数据
                await self._collect_data_from_routers()
                
                # 计算休眠时间，以维持轮询间隔
                elapsed = time.time() - start_time
                sleep_time = max(0, self.settings.hardware_polling_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Data collection loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in data collection loop: {e}")
            self.last_error = str(e)
    
    async def _monitoring_loop(self):
        """用于硬件监控的后台循环。"""
        try:
            while self.is_running:
                # 监控路由器连接状态
                await self._monitor_router_health()
                
                # 更新统计信息
                self._update_sample_rate_stats()
                
                # 等待下一次检查
                await asyncio.sleep(30)  # 每 30 秒检查一次
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_data_from_routers(self):
        """从所有已连接的路由器采集 CSI 数据。"""
        for router_id, interface in self.router_interfaces.items():
            try:
                # 从路由器获取 CSI 数据
                csi_data = await interface.get_csi_data()
                
                if csi_data is not None:
                    # 处理采集到的数据
                    await self._process_collected_data(router_id, csi_data)
                    
                    self.stats["successful_samples"] += 1
                    self.stats["last_sample_time"] = datetime.now().isoformat()
                else:
                    self.stats["failed_samples"] += 1
                
                self.stats["total_samples"] += 1
                
            except Exception as e:
                self.logger.error(f"Error collecting data from router {router_id}: {e}")
                self.stats["failed_samples"] += 1
                self.stats["total_samples"] += 1
    
    async def _process_collected_data(self, router_id: str, csi_data: np.ndarray):
        """处理采集到的 CSI 数据。"""
        try:
            # 创建样本元数据
            metadata = {
                "router_id": router_id,
                "timestamp": datetime.now().isoformat(),
                "sample_rate": self.stats["average_sample_rate"],
                "data_shape": csi_data.shape if hasattr(csi_data, 'shape') else None
            }
            
            # 加入最近样本缓冲区
            sample = {
                "router_id": router_id,
                "timestamp": metadata["timestamp"],
                "data": csi_data,
                "metadata": metadata
            }
            
            self.recent_samples.append(sample)
            
            # 维护缓冲区大小
            if len(self.recent_samples) > self.max_recent_samples:
                self.recent_samples.pop(0)
            
            # 通知其他服务（通常应通过事件系统完成）
            # 当前实现中仅记录采集日志
            self.logger.debug(f"Collected CSI data from {router_id}: shape {csi_data.shape if hasattr(csi_data, 'shape') else 'unknown'}")
            
        except Exception as e:
            self.logger.error(f"Error processing collected data: {e}")
    
    async def _monitor_router_health(self):
        """监控路由器连接健康状态。"""
        healthy_routers = 0
        
        for router_id, interface in self.router_interfaces.items():
            try:
                is_healthy = await interface.check_health()
                
                if is_healthy:
                    healthy_routers += 1
                else:
                    self.logger.warning(f"Router {router_id} is unhealthy")
                    
                    # 若不处于模拟模式，则尝试重连
                    if not self.settings.mock_hardware:
                        try:
                            await interface.reconnect()
                            self.logger.info(f"Reconnected to router {router_id}")
                        except Exception as e:
                            self.logger.error(f"Failed to reconnect to router {router_id}: {e}")
                
            except Exception as e:
                self.logger.error(f"Error checking health of router {router_id}: {e}")
        
        self.stats["connected_routers"] = healthy_routers
    
    def _update_sample_rate_stats(self):
        """更新采样率统计信息。"""
        if len(self.recent_samples) < 2:
            return
        
        # 根据最近样本计算采样率
        recent_count = min(100, len(self.recent_samples))
        recent_samples = self.recent_samples[-recent_count:]
        
        if len(recent_samples) >= 2:
            # 计算时间差
            time_diffs = []
            for i in range(1, len(recent_samples)):
                try:
                    t1 = datetime.fromisoformat(recent_samples[i-1]["timestamp"])
                    t2 = datetime.fromisoformat(recent_samples[i]["timestamp"])
                    diff = (t2 - t1).total_seconds()
                    if diff > 0:
                        time_diffs.append(diff)
                except Exception:
                    continue
            
            if time_diffs:
                avg_interval = sum(time_diffs) / len(time_diffs)
                self.stats["average_sample_rate"] = 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    async def get_router_status(self, router_id: str) -> Dict[str, Any]:
        """获取指定路由器的状态。"""
        if router_id not in self.router_interfaces:
            raise ValueError(f"Router {router_id} not found")
        
        interface = self.router_interfaces[router_id]
        
        try:
            is_healthy = await interface.check_health()
            status = await interface.get_status()
            
            return {
                "router_id": router_id,
                "healthy": is_healthy,
                "connected": status.get("connected", False),
                "last_data_time": status.get("last_data_time"),
                "error_count": status.get("error_count", 0),
                "configuration": status.get("configuration", {})
            }
            
        except Exception as e:
            return {
                "router_id": router_id,
                "healthy": False,
                "connected": False,
                "error": str(e)
            }
    
    async def get_all_router_status(self) -> List[Dict[str, Any]]:
        """获取所有路由器的状态。"""
        statuses = []
        
        for router_id in self.router_interfaces:
            try:
                status = await self.get_router_status(router_id)
                statuses.append(status)
            except Exception as e:
                statuses.append({
                    "router_id": router_id,
                    "healthy": False,
                    "error": str(e)
                })
        
        return statuses
    
    async def get_recent_data(self, router_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的 CSI 数据样本。"""
        samples = self.recent_samples[-limit:] if limit else self.recent_samples
        
        if router_id:
            samples = [s for s in samples if s["router_id"] == router_id]
        
        # 将 numpy 数组转换为列表，以便 JSON 序列化
        result = []
        for sample in samples:
            sample_copy = sample.copy()
            if isinstance(sample_copy["data"], np.ndarray):
                sample_copy["data"] = sample_copy["data"].tolist()
            result.append(sample_copy)
        
        return result
    
    async def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "status": "healthy" if self.is_running and not self.last_error else "unhealthy",
            "running": self.is_running,
            "last_error": self.last_error,
            "statistics": self.stats.copy(),
            "configuration": {
                "mock_hardware": self.settings.mock_hardware,
                "wifi_interface": self.settings.wifi_interface,
                "polling_interval": self.settings.hardware_polling_interval,
                "buffer_size": self.settings.csi_buffer_size
            },
            "routers": await self.get_all_router_status()
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标。"""
        total_samples = self.stats["total_samples"]
        success_rate = self.stats["successful_samples"] / max(1, total_samples)
        
        return {
            "hardware_service": {
                "total_samples": total_samples,
                "successful_samples": self.stats["successful_samples"],
                "failed_samples": self.stats["failed_samples"],
                "success_rate": success_rate,
                "average_sample_rate": self.stats["average_sample_rate"],
                "connected_routers": self.stats["connected_routers"],
                "last_sample_time": self.stats["last_sample_time"]
            }
        }
    
    async def reset(self):
        """重置服务状态。"""
        self.stats = {
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "average_sample_rate": 0.0,
            "last_sample_time": None,
            "connected_routers": len(self.router_interfaces)
        }
        
        self.recent_samples.clear()
        self.last_error = None
        
        self.logger.info("Hardware service reset")
    
    async def trigger_manual_collection(self, router_id: Optional[str] = None) -> Dict[str, Any]:
        """手动触发数据采集。"""
        if not self.is_running:
            raise RuntimeError("Hardware service is not running")
        
        results = {}
        
        if router_id:
            # 从指定路由器采集
            if router_id not in self.router_interfaces:
                raise ValueError(f"Router {router_id} not found")
            
            interface = self.router_interfaces[router_id]
            try:
                csi_data = await interface.get_csi_data()
                if csi_data is not None:
                    await self._process_collected_data(router_id, csi_data)
                    results[router_id] = {"success": True, "data_shape": csi_data.shape if hasattr(csi_data, 'shape') else None}
                else:
                    results[router_id] = {"success": False, "error": "No data received"}
            except Exception as e:
                results[router_id] = {"success": False, "error": str(e)}
        else:
            # 从所有路由器采集
            await self._collect_data_from_routers()
            results = {"message": "Manual collection triggered for all routers"}
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """执行健康检查。"""
        try:
            status = "healthy" if self.is_running and not self.last_error else "unhealthy"
            
            # 检查路由器健康状态
            healthy_routers = 0
            total_routers = len(self.router_interfaces)
            
            for router_id, interface in self.router_interfaces.items():
                try:
                    if await interface.check_health():
                        healthy_routers += 1
                except Exception:
                    pass
            
            return {
                "status": status,
                "message": self.last_error if self.last_error else "Hardware service is running normally",
                "connected_routers": f"{healthy_routers}/{total_routers}",
                "metrics": {
                    "total_samples": self.stats["total_samples"],
                    "success_rate": (
                        self.stats["successful_samples"] / max(1, self.stats["total_samples"])
                    ),
                    "average_sample_rate": self.stats["average_sample_rate"]
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}"
            }
    
    async def is_ready(self) -> bool:
        """检查服务是否就绪。"""
        return self.is_running and len(self.router_interfaces) > 0
