"""
用于 WiFi CSI 数据采集的路由器接口
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class RouterInterface:
    """用于连接 WiFi 路由器并采集 CSI 数据的接口。"""
    
    def __init__(
        self,
        router_id: str,
        host: str,
        port: int = 22,
        username: str = "admin",
        password: str = "",
        interface: str = "wlan0",
        mock_mode: bool = False
    ):
        """初始化路由器接口。
        
        Args:
            router_id: 路由器的唯一标识符
            host: 路由器 IP 地址或主机名
            port: 连接使用的 SSH 端口
            username: SSH 用户名
            password: SSH 密码
            interface: WiFi 接口名称
            mock_mode: 是否使用模拟数据而不是真实连接
        """
        self.router_id = router_id
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.interface = interface
        self.mock_mode = mock_mode
        
        self.logger = logging.getLogger(f"{__name__}.{router_id}")
        
        # 连接状态
        self.is_connected = False
        self.connection = None
        self.last_error = None
        
        # 数据采集状态
        self.last_data_time = None
        self.error_count = 0
        self.sample_count = 0
        
        # 模拟数据生成（委托给 testing 模块）
        self._mock_csi_generator = None
        if mock_mode:
            self._initialize_mock_generator()

    def _initialize_mock_generator(self):
        """从 testing 模块初始化模拟数据生成器。"""
        from src.testing.mock_csi_generator import MockCSIGenerator
        self._mock_csi_generator = MockCSIGenerator()
        self._mock_csi_generator.show_banner()
    
    async def connect(self):
        """连接到路由器。"""
        if self.mock_mode:
            self.is_connected = True
            self.logger.info(f"Mock connection established to router {self.router_id}")
            return
        
        try:
            self.logger.info(f"Connecting to router {self.router_id} at {self.host}:{self.port}")
            
            # 在真实实现中，这里会建立 SSH 连接
            # 当前先模拟连接过程
            await asyncio.sleep(0.1)  # 模拟连接延迟
            
            self.is_connected = True
            self.error_count = 0
            self.logger.info(f"Connected to router {self.router_id}")
            
        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            self.logger.error(f"Failed to connect to router {self.router_id}: {e}")
            raise
    
    async def disconnect(self):
        """断开与路由器的连接。"""
        try:
            if self.connection:
                # 关闭 SSH 连接
                self.connection = None
            
            self.is_connected = False
            self.logger.info(f"Disconnected from router {self.router_id}")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from router {self.router_id}: {e}")
    
    async def reconnect(self):
        """重新连接路由器。"""
        await self.disconnect()
        await asyncio.sleep(1)  # 重连前等待
        await self.connect()
    
    async def get_csi_data(self) -> Optional[np.ndarray]:
        """从路由器获取 CSI 数据。
        
        Returns:
            以 numpy 数组形式返回 CSI 数据；若无数据则返回 None
        """
        if not self.is_connected:
            raise RuntimeError(f"Router {self.router_id} is not connected")
        
        try:
            if self.mock_mode:
                csi_data = self._generate_mock_csi_data()
            else:
                csi_data = await self._collect_real_csi_data()
            
            if csi_data is not None:
                self.last_data_time = datetime.now()
                self.sample_count += 1
                self.error_count = 0
            
            return csi_data
            
        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            self.logger.error(f"Error getting CSI data from router {self.router_id}: {e}")
            return None
    
    def _generate_mock_csi_data(self) -> np.ndarray:
        """生成用于测试的模拟 CSI 数据。

        委托给 testing 模块中的 MockCSIGenerator。
        该方法仅在 `mock_mode=True` 时可调用。
        """
        if self._mock_csi_generator is None:
            self._initialize_mock_generator()
        return self._mock_csi_generator.generate()
    
    async def _collect_real_csi_data(self) -> Optional[np.ndarray]:
        """从路由器采集真实 CSI 数据。

        Raises:
            RuntimeError: 在当前状态下总会抛出，因为真实 CSI
                数据采集依赖尚未配置的硬件环境。该方法绝不能静默返回
                随机数据或占位数据。
        """
        raise RuntimeError(
            f"Real CSI data collection from router '{self.router_id}' requires "
            "hardware setup that is not configured. You must: "
            "(1) install CSI-capable firmware (e.g., Atheros CSI Tool, Nexmon CSI) on the router, "
            "(2) configure the SSH connection to the router, and "
            "(3) implement the CSI extraction command for your specific firmware. "
            "For development/testing, use mock_mode=True. "
            "See docs/hardware-setup.md for complete setup instructions."
        )
    
    async def check_health(self) -> bool:
        """检查路由器连接是否健康。
        
        Returns:
            健康时返回 True，否则返回 False
        """
        if not self.is_connected:
            return False
        
        try:
            # 在模拟模式下始终视为健康
            if self.mock_mode:
                return True
            
            # 对真实连接，可通过 ping 路由器或检查 SSH 连接状态来判定
            # 当前实现中，若错误次数较少则视为健康
            return self.error_count < 5
            
        except Exception as e:
            self.logger.error(f"Error checking health of router {self.router_id}: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取路由器状态信息。
        
        Returns:
            包含路由器状态的字典
        """
        return {
            "router_id": self.router_id,
            "connected": self.is_connected,
            "mock_mode": self.mock_mode,
            "last_data_time": self.last_data_time.isoformat() if self.last_data_time else None,
            "error_count": self.error_count,
            "sample_count": self.sample_count,
            "last_error": self.last_error,
            "configuration": {
                "host": self.host,
                "port": self.port,
                "username": self.username,
                "interface": self.interface
            }
        }
    
    async def get_router_info(self) -> Dict[str, Any]:
        """获取路由器硬件信息。
        
        Returns:
            包含路由器信息的字典
        """
        if self.mock_mode:
            if self._mock_csi_generator is None:
                self._initialize_mock_generator()
            return self._mock_csi_generator.get_router_info()
        
        # 对真实路由器，这里应查询实际硬件信息
        return {
            "model": "Unknown",
            "firmware": "Unknown",
            "wifi_standard": "Unknown",
            "antennas": 1,
            "supported_bands": ["Unknown"],
            "csi_capabilities": {
                "max_subcarriers": 64,
                "max_antennas": 1,
                "sampling_rate": 100
            }
        }
    
    async def configure_csi_collection(self, config: Dict[str, Any]) -> bool:
        """配置 CSI 数据采集参数。
        
        Args:
            config: 配置字典
            
        Returns:
            配置成功返回 True，否则返回 False
        """
        try:
            if self.mock_mode:
                if self._mock_csi_generator is None:
                    self._initialize_mock_generator()
                self._mock_csi_generator.configure(config)
                self.logger.info(f"Mock CSI collection configured for router {self.router_id}")
                return True
            
            # 对真实路由器，这里应发送配置命令
            self.logger.warning("Real CSI configuration not implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Error configuring CSI collection for router {self.router_id}: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取路由器接口指标。
        
        Returns:
            包含指标信息的字典
        """
        uptime = 0
        if self.last_data_time:
            uptime = (datetime.now() - self.last_data_time).total_seconds()
        
        success_rate = 0
        if self.sample_count > 0:
            success_rate = (self.sample_count - self.error_count) / self.sample_count
        
        return {
            "router_id": self.router_id,
            "sample_count": self.sample_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "uptime_seconds": uptime,
            "is_connected": self.is_connected,
            "mock_mode": self.mock_mode
        }
    
    def reset_stats(self):
        """重置统计计数器。"""
        self.error_count = 0
        self.sample_count = 0
        self.last_error = None
        self.logger.info(f"Statistics reset for router {self.router_id}")
