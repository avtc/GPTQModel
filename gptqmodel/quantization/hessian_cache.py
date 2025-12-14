# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import threading
from typing import Dict, List, Optional, Tuple

import torch


class HessianCache:
    """
    A global, thread-safe cache pool for Hessian accumulators to reduce memory allocation overhead.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(HessianCache, cls).__new__(cls)
                    cls._instance.cache: Dict[str, List[torch.Tensor]] = {}
                    cls._instance.locks: Dict[Tuple[str, Optional[int]], threading.Lock] = {}
        return cls._instance

    def _get_device_lock(self, device: torch.device) -> threading.Lock:
        key = (device.type, device.index)
        with self._lock:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            return self.locks[key]

    def get(self, shape: torch.Size, device: torch.device) -> Optional[torch.Tensor]:
        """
        Retrieves a tensor of the specified shape and device from the cache pool in a thread-safe manner.

        Args:
            shape (torch.Size): The desired tensor shape.
            device (torch.device): The desired tensor device.

        Returns:
            Optional[torch.Tensor]: A cached tensor if available, otherwise None.
        """
        device_lock = self._get_device_lock(device)
        with device_lock:
            key = f"{shape}-{device}"
            print(f"DEBUG: GET key={key} cache_len={len(self.cache.get(key, []))}")
            if key in self.cache and self.cache[key]:
                return self.cache[key].pop()
            return None

    def put(self, tensor: torch.Tensor):
        """
        Adds a tensor back to the cache pool for future reuse in a thread-safe manner.

        Args:
            tensor (torch.Tensor): The tensor to cache.
        """
        device_lock = self._get_device_lock(tensor.device)
        with device_lock:
            key = f"{tensor.shape}-{tensor.device}"
            print(f"DEBUG: PUT key={key}")
            if key not in self.cache:
                self.cache[key] = []
            self.cache[key].append(tensor)
