"""
WiFi-DensePose API 的主服务编排器
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from src.config.settings import Settings
from src.services.health_check import HealthCheckService
from src.services.metrics import MetricsService
from src.api.dependencies import (
    get_hardware_service,
    get_pose_service,
    get_stream_service
)
from src.api.websocket.connection_manager import connection_manager
from src.api.websocket.pose_stream import PoseStreamHandler

logger = logging.getLogger(__name__)


class ServiceOrchestrator:
    """负责管理所有应用服务的主服务编排器。"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._services: Dict[str, Any] = {}
        self._background_tasks: List[asyncio.Task] = []
        self._initialized = False
        self._started = False
        
        # 核心服务
        self.health_service = HealthCheckService(settings)
        self.metrics_service = MetricsService(settings)
        
        # 应用服务（稍后初始化）
        self.hardware_service = None
        self.pose_service = None
        self.stream_service = None
        self.pose_stream_handler = None
    
    async def initialize(self):
        """初始化所有服务。"""
        if self._initialized:
            logger.warning("Services already initialized")
            return
        
        logger.info("Initializing services...")
        
        try:
            # 初始化核心服务
            await self.health_service.initialize()
            await self.metrics_service.initialize()
            
            # 初始化应用服务
            await self._initialize_application_services()
            
            # 将服务存入注册表
            self._services = {
                'health': self.health_service,
                'metrics': self.metrics_service,
                'hardware': self.hardware_service,
                'pose': self.pose_service,
                'stream': self.stream_service,
                'pose_stream_handler': self.pose_stream_handler,
                'connection_manager': connection_manager
            }
            
            self._initialized = True
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            await self.shutdown()
            raise
    
    async def _initialize_application_services(self):
        """初始化应用特定服务。"""
        try:
            # 初始化硬件服务
            self.hardware_service = get_hardware_service()
            await self.hardware_service.initialize()
            logger.info("Hardware service initialized")
            
            # 初始化姿态服务
            self.pose_service = get_pose_service()
            await self.pose_service.initialize()
            logger.info("Pose service initialized")
            
            # 初始化流服务
            self.stream_service = get_stream_service()
            await self.stream_service.initialize()
            logger.info("Stream service initialized")
            
            # 初始化姿态流处理器
            self.pose_stream_handler = PoseStreamHandler(
                connection_manager=connection_manager,
                pose_service=self.pose_service,
                stream_service=self.stream_service
            )
            logger.info("Pose stream handler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize application services: {e}")
            raise
    
    async def start(self):
        """启动所有服务和后台任务。"""
        if not self._initialized:
            await self.initialize()
        
        if self._started:
            logger.warning("Services already started")
            return
        
        logger.info("Starting services...")
        
        try:
            # 启动核心服务
            await self.health_service.start()
            await self.metrics_service.start()
            
            # 启动应用服务
            await self._start_application_services()
            
            # 启动后台任务
            await self._start_background_tasks()
            
            self._started = True
            logger.info("All services started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            await self.shutdown()
            raise
    
    async def _start_application_services(self):
        """启动应用特定服务。"""
        try:
            # 启动硬件服务
            if hasattr(self.hardware_service, 'start'):
                await self.hardware_service.start()
            
            # 启动姿态服务
            if hasattr(self.pose_service, 'start'):
                await self.pose_service.start()
            
            # 启动流服务
            if hasattr(self.stream_service, 'start'):
                await self.stream_service.start()
            
            logger.info("Application services started")
            
        except Exception as e:
            logger.error(f"Failed to start application services: {e}")
            raise
    
    async def _start_background_tasks(self):
        """启动后台任务。"""
        try:
            # 启动健康检查监控
            if self.settings.health_check_interval > 0:
                task = asyncio.create_task(self._health_check_loop())
                self._background_tasks.append(task)
            
            # 启动指标采集
            if self.settings.metrics_enabled:
                task = asyncio.create_task(self._metrics_collection_loop())
                self._background_tasks.append(task)
            
            # 若启用，则启动姿态流推送
            if self.settings.enable_real_time_processing:
                await self.pose_stream_handler.start_streaming()
            
            logger.info(f"Started {len(self._background_tasks)} background tasks")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise
    
    async def _health_check_loop(self):
        """后台健康检查循环。"""
        logger.info("Starting health check loop")
        
        while True:
            try:
                await self.health_service.perform_health_checks()
                await asyncio.sleep(self.settings.health_check_interval)
            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.settings.health_check_interval)
    
    async def _metrics_collection_loop(self):
        """后台指标采集循环。"""
        logger.info("Starting metrics collection loop")
        
        while True:
            try:
                await self.metrics_service.collect_metrics()
                await asyncio.sleep(60)  # 每分钟采集一次指标
            except asyncio.CancelledError:
                logger.info("Metrics collection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)
    
    async def shutdown(self):
        """关闭所有服务并清理资源。"""
        logger.info("Shutting down services...")
        
        try:
            # 取消后台任务
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
                self._background_tasks.clear()
            
            # 停止姿态流推送
            if self.pose_stream_handler:
                await self.pose_stream_handler.shutdown()
            
            # 关闭连接管理器
            await connection_manager.shutdown()
            
            # 关闭应用服务
            await self._shutdown_application_services()
            
            # 关闭核心服务
            await self.health_service.shutdown()
            await self.metrics_service.shutdown()
            
            self._started = False
            self._initialized = False
            
            logger.info("All services shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _shutdown_application_services(self):
        """关闭应用特定服务。"""
        try:
            # 按相反顺序关闭服务
            if self.stream_service and hasattr(self.stream_service, 'shutdown'):
                await self.stream_service.shutdown()
            
            if self.pose_service and hasattr(self.pose_service, 'shutdown'):
                await self.pose_service.shutdown()
            
            if self.hardware_service and hasattr(self.hardware_service, 'shutdown'):
                await self.hardware_service.shutdown()
            
            logger.info("Application services shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down application services: {e}")
    
    async def restart_service(self, service_name: str):
        """重启指定服务。"""
        logger.info(f"Restarting service: {service_name}")
        
        service = self._services.get(service_name)
        if not service:
            raise ValueError(f"Service not found: {service_name}")
        
        try:
            # 停止服务
            if hasattr(service, 'stop'):
                await service.stop()
            elif hasattr(service, 'shutdown'):
                await service.shutdown()
            
            # 重新初始化服务
            if hasattr(service, 'initialize'):
                await service.initialize()
            
            # 启动服务
            if hasattr(service, 'start'):
                await service.start()
            
            logger.info(f"Service restarted successfully: {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            raise
    
    async def reset_services(self):
        """将所有服务重置为初始状态。"""
        logger.info("Resetting all services")
        
        try:
            # 重置应用服务
            if self.hardware_service and hasattr(self.hardware_service, 'reset'):
                await self.hardware_service.reset()
            
            if self.pose_service and hasattr(self.pose_service, 'reset'):
                await self.pose_service.reset()
            
            if self.stream_service and hasattr(self.stream_service, 'reset'):
                await self.stream_service.reset()
            
            # 重置连接管理器
            await connection_manager.reset()
            
            logger.info("All services reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset services: {e}")
            raise
    
    async def get_service_status(self) -> Dict[str, Any]:
        """获取所有服务的状态。"""
        status = {}
        
        for name, service in self._services.items():
            try:
                if hasattr(service, 'get_status'):
                    status[name] = await service.get_status()
                else:
                    status[name] = {"status": "unknown"}
            except Exception as e:
                status[name] = {"status": "error", "error": str(e)}
        
        return status
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """获取所有服务的指标。"""
        metrics = {}
        
        for name, service in self._services.items():
            try:
                if hasattr(service, 'get_metrics'):
                    metrics[name] = await service.get_metrics()
                elif hasattr(service, 'get_performance_metrics'):
                    metrics[name] = await service.get_performance_metrics()
            except Exception as e:
                logger.error(f"Failed to get metrics from {name}: {e}")
                metrics[name] = {"error": str(e)}
        
        return metrics
    
    async def get_service_info(self) -> Dict[str, Any]:
        """获取所有服务的信息。"""
        info = {
            "total_services": len(self._services),
            "initialized": self._initialized,
            "started": self._started,
            "background_tasks": len(self._background_tasks),
            "services": {}
        }
        
        for name, service in self._services.items():
            service_info = {
                "type": type(service).__name__,
                "module": type(service).__module__
            }
            
            # 若可用，则补充服务特定信息
            if hasattr(service, 'get_info'):
                try:
                    service_info.update(await service.get_info())
                except Exception as e:
                    service_info["error"] = str(e)
            
            info["services"][name] = service_info
        
        return info
    
    def get_service(self, name: str) -> Optional[Any]:
        """按名称获取指定服务。"""
        return self._services.get(name)
    
    @property
    def is_healthy(self) -> bool:
        """检查所有服务是否健康。"""
        return self._initialized and self._started
    
    @asynccontextmanager
    async def service_context(self):
        """用于服务生命周期管理的上下文管理器。"""
        try:
            await self.initialize()
            await self.start()
            yield self
        finally:
            await self.shutdown()
