import os
import time
import shutil
import logging
import threading
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import psutil
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LatencySummary:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    total_time_ms: float
    qps: float


@dataclass
class ResourceProfile:
    # Latency
    latency: LatencySummary
    # CPU
    avg_process_cpu_percent: float
    peak_process_cpu_percent: float
    avg_system_cpu_percent: float
    peak_system_cpu_percent: float
    # GPU
    gpu_available: bool
    gpu_name: str
    avg_gpu_utilization_percent: float
    peak_gpu_utilization_percent: float
    avg_gpu_memory_used_mb: float
    peak_gpu_memory_used_mb: float
    gpu_memory_delta_mb: float
    # RAM
    baseline_ram_mb: float
    peak_ram_mb: float
    final_ram_mb: float
    ram_delta_mb: float
    # Storage
    storage_footprint: Dict[str, Any] = field(default_factory=dict)


class ResourceSampler:
    """
    Background sampling thread to monitor CPU, RAM, and GPU usage
    during query workload execution without blocking the main search loop.
    """

    def __init__(self, sample_interval: float = 0.05):
        self.interval = sample_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.proc = psutil.Process()

        self.proc_cpu_samples: List[float] = []
        self.sys_cpu_samples: List[float] = []
        self.ram_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.gpu_mem_samples: List[float] = []

        self.gpu_available = False
        self.gpu_name = "N/A"
        self._detect_gpu()

        self.initial_ram_mb = 0.0
        self.initial_gpu_mem_mb = 0.0

    def _detect_gpu(self) -> None:
        """Detect GPU availability and name using nvidia-smi."""
        try:
            cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            out = subprocess.check_output(
                cmd, text=True, stderr=subprocess.DEVNULL, creationflags=0x08000000
            ).strip()
            if out:
                self.gpu_available = True
                self.gpu_name = out.split("\n")[0].strip()
        except Exception:
            self.gpu_available = False
            self.gpu_name = "N/A - CPU only"

    def _get_gpu_snapshot(self) -> Optional[Tuple[float, float]]:
        """Query instantaneous GPU utilization % and memory used (MiB)."""
        if not self.gpu_available:
            return None
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(
                cmd, text=True, stderr=subprocess.DEVNULL, creationflags=0x08000000
            ).strip()
            if out:
                parts = [float(x.strip()) for x in out.split(",")]
                return parts[0], parts[1]
        except Exception:
            pass
        return None

    def _run_sampling(self) -> None:
        # Prime psutil cpu counters
        try:
            self.proc.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while not self._stop_event.is_set():
            try:
                proc_cpu = self.proc.cpu_percent(interval=None)
                sys_cpu = psutil.cpu_percent(interval=None)
                ram_mb = self.proc.memory_info().rss / (1024.0 * 1024.0)

                self.proc_cpu_samples.append(proc_cpu)
                self.sys_cpu_samples.append(sys_cpu)
                self.ram_samples.append(ram_mb)

                if self.gpu_available:
                    gpu_snap = self._get_gpu_snapshot()
                    if gpu_snap is not None:
                        self.gpu_util_samples.append(gpu_snap[0])
                        self.gpu_mem_samples.append(gpu_snap[1])
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self) -> None:
        """Start background sampling."""
        self.proc_cpu_samples.clear()
        self.sys_cpu_samples.clear()
        self.ram_samples.clear()
        self.gpu_util_samples.clear()
        self.gpu_mem_samples.clear()
        self._stop_event.clear()

        self.initial_ram_mb = self.proc.memory_info().rss / (1024.0 * 1024.0)
        gpu_snap = self._get_gpu_snapshot()
        self.initial_gpu_mem_mb = gpu_snap[1] if gpu_snap else 0.0

        self._thread = threading.Thread(target=self._run_sampling, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """
        Stop sampling and return aggregated metrics:
        (avg_proc_cpu, peak_proc_cpu, avg_sys_cpu, peak_sys_cpu,
         avg_gpu_util, peak_gpu_util, avg_gpu_mem, peak_gpu_mem, gpu_mem_delta)
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        # CPU aggregates
        avg_proc_cpu = (
            sum(self.proc_cpu_samples) / len(self.proc_cpu_samples)
            if self.proc_cpu_samples
            else 0.0
        )
        peak_proc_cpu = max(self.proc_cpu_samples, default=0.0)

        avg_sys_cpu = (
            sum(self.sys_cpu_samples) / len(self.sys_cpu_samples)
            if self.sys_cpu_samples
            else 0.0
        )
        peak_sys_cpu = max(self.sys_cpu_samples, default=0.0)

        # GPU aggregates
        avg_gpu_util = (
            sum(self.gpu_util_samples) / len(self.gpu_util_samples)
            if self.gpu_util_samples
            else 0.0
        )
        peak_gpu_util = max(self.gpu_util_samples, default=0.0)

        avg_gpu_mem = (
            sum(self.gpu_mem_samples) / len(self.gpu_mem_samples)
            if self.gpu_mem_samples
            else self.initial_gpu_mem_mb
        )
        peak_gpu_mem = max(self.gpu_mem_samples, default=self.initial_gpu_mem_mb)
        gpu_mem_delta = max(0.0, peak_gpu_mem - self.initial_gpu_mem_mb)

        return (
            round(avg_proc_cpu, 2),
            round(peak_proc_cpu, 2),
            round(avg_sys_cpu, 2),
            round(peak_sys_cpu, 2),
            round(avg_gpu_util, 2),
            round(peak_gpu_util, 2),
            round(avg_gpu_mem, 2),
            round(peak_gpu_mem, 2),
            round(gpu_mem_delta, 2),
        )


class StorageAuditor:
    """Audits disk index size, model weights footprint, and database vector storage."""

    @staticmethod
    def get_file_size_bytes(file_path: str | Path) -> int:
        p = Path(file_path)
        return p.stat().st_size if p.exists() else 0

    @staticmethod
    def get_directory_size_bytes(dir_path: str | Path) -> int:
        p = Path(dir_path)
        if not p.exists() or not p.is_dir():
            return 0
        total = 0
        for entry in p.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except Exception:
                    pass
        return total

    @classmethod
    def get_hf_model_size_bytes(cls, model_name: str = "BAAI/bge-m3") -> int:
        """Find HuggingFace cache folder for the specified model and sum its size."""
        hf_home = os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub")
        )
        hub_path = Path(hf_home)
        sanitized = "models--" + model_name.replace("/", "--")
        target_dir = hub_path / sanitized
        if target_dir.exists():
            return cls.get_directory_size_bytes(target_dir)

        # Fallback: scan for any directory matching model name
        for d in hub_path.glob(f"*{model_name.split('/')[-1]}*"):
            if d.is_dir():
                return cls.get_directory_size_bytes(d)
        return 0

    @staticmethod
    def get_db_vector_storage_bytes(db: Session) -> Tuple[int, int]:
        """
        Query PostgreSQL for pgvector column storage and total relation size.
        Returns (embedding_column_bytes, total_table_bytes).
        """
        try:
            res = db.execute(
                text(
                    "SELECT sum(pg_column_size(embedding)) as emb_bytes, "
                    "pg_total_relation_size('lexicons') as table_bytes "
                    "FROM lexicons"
                )
            ).fetchone()
            if res:
                emb_bytes = int(res[0] or 0)
                tbl_bytes = int(res[1] or 0)
                return emb_bytes, tbl_bytes
        except Exception as e:
            logger.warning(f"Could not query PostgreSQL storage size: {e}")
        return 0, 0

    @classmethod
    def audit_method_storage(cls, method_name: str, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Produce a comprehensive storage breakdown for a given search method.
        """
        bm25_bytes = cls.get_file_size_bytes(settings.INDEX_CACHE_PATH)
        hf_model_bytes = cls.get_hf_model_size_bytes(settings.EMBEDDING_MODEL_NAME)
        emb_bytes, table_bytes = cls.get_db_vector_storage_bytes(db) if db else (0, 0)

        method = method_name.lower()
        if "bm25" in method:
            return {
                "method": "BM25",
                "bm25_index_file_bytes": bm25_bytes,
                "bm25_index_file_kb": round(bm25_bytes / 1024.0, 2),
                "model_weights_bytes": 0,
                "model_weights_mb": 0.0,
                "db_vector_column_bytes": 0,
                "db_vector_column_kb": 0.0,
                "total_footprint_mb": round(bm25_bytes / (1024.0 * 1024.0), 4),
            }
        elif "bge" in method or "semantic" in method:
            total_mb = (hf_model_bytes + emb_bytes) / (1024.0 * 1024.0)
            return {
                "method": "BGE-M3",
                "bm25_index_file_bytes": 0,
                "bm25_index_file_kb": 0.0,
                "model_weights_bytes": hf_model_bytes,
                "model_weights_mb": round(hf_model_bytes / (1024.0 * 1024.0), 2),
                "db_vector_column_bytes": emb_bytes,
                "db_vector_column_kb": round(emb_bytes / 1024.0, 2),
                "db_table_total_bytes": table_bytes,
                "db_table_total_mb": round(table_bytes / (1024.0 * 1024.0), 2),
                "total_footprint_mb": round(total_mb, 2),
            }
        else:  # Hybrid
            total_mb = (bm25_bytes + hf_model_bytes + emb_bytes) / (1024.0 * 1024.0)
            return {
                "method": "Hybrid Search",
                "bm25_index_file_bytes": bm25_bytes,
                "bm25_index_file_kb": round(bm25_bytes / 1024.0, 2),
                "model_weights_bytes": hf_model_bytes,
                "model_weights_mb": round(hf_model_bytes / (1024.0 * 1024.0), 2),
                "db_vector_column_bytes": emb_bytes,
                "db_vector_column_kb": round(emb_bytes / 1024.0, 2),
                "db_table_total_bytes": table_bytes,
                "db_table_total_mb": round(table_bytes / (1024.0 * 1024.0), 2),
                "total_footprint_mb": round(total_mb, 2),
            }


class ResourceMonitor:
    """Orchestrates resource sampling, timing, and storage auditing."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.sampler = ResourceSampler()

    def start_monitoring(self) -> None:
        self.sampler.start()

    def stop_monitoring(
        self,
        latencies_ms: List[float],
        method_name: str,
    ) -> ResourceProfile:
        (
            avg_proc_cpu,
            peak_proc_cpu,
            avg_sys_cpu,
            peak_sys_cpu,
            avg_gpu_util,
            peak_gpu_util,
            avg_gpu_mem,
            peak_gpu_mem,
            gpu_mem_delta,
        ) = self.sampler.stop()

        # RAM
        proc = psutil.Process()
        final_ram_mb = proc.memory_info().rss / (1024.0 * 1024.0)
        peak_ram_mb = max(self.sampler.ram_samples, default=final_ram_mb)
        ram_delta_mb = max(0.0, peak_ram_mb - self.sampler.initial_ram_mb)

        # Latency statistics
        if latencies_ms:
            sorted_lat = sorted(latencies_ms)
            n = len(sorted_lat)
            mean_ms = sum(sorted_lat) / n
            median_ms = sorted_lat[n // 2]
            p95_idx = min(int(n * 0.95), n - 1)
            p99_idx = min(int(n * 0.99), n - 1)
            p95_ms = sorted_lat[p95_idx]
            p99_ms = sorted_lat[p99_idx]
            min_ms = sorted_lat[0]
            max_ms = sorted_lat[-1]
            variance = sum((x - mean_ms) ** 2 for x in sorted_lat) / n
            std_ms = variance ** 0.5
            total_time_ms = sum(sorted_lat)
            qps = (n / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0
        else:
            mean_ms = median_ms = p95_ms = p99_ms = min_ms = max_ms = std_ms = total_time_ms = qps = 0.0
            n = 0

        latency_summary = LatencySummary(
            count=n,
            mean_ms=round(mean_ms, 3),
            median_ms=round(median_ms, 3),
            p95_ms=round(p95_ms, 3),
            p99_ms=round(p99_ms, 3),
            min_ms=round(min_ms, 3),
            max_ms=round(max_ms, 3),
            std_ms=round(std_ms, 3),
            total_time_ms=round(total_time_ms, 3),
            qps=round(qps, 2),
        )

        storage = StorageAuditor.audit_method_storage(method_name, self.db)

        return ResourceProfile(
            latency=latency_summary,
            avg_process_cpu_percent=avg_proc_cpu,
            peak_process_cpu_percent=peak_proc_cpu,
            avg_system_cpu_percent=avg_sys_cpu,
            peak_system_cpu_percent=peak_sys_cpu,
            gpu_available=self.sampler.gpu_available,
            gpu_name=self.sampler.gpu_name,
            avg_gpu_utilization_percent=avg_gpu_util,
            peak_gpu_utilization_percent=peak_gpu_util,
            avg_gpu_memory_used_mb=avg_gpu_mem,
            peak_gpu_memory_used_mb=peak_gpu_mem,
            gpu_memory_delta_mb=gpu_mem_delta,
            baseline_ram_mb=round(self.sampler.initial_ram_mb, 2),
            peak_ram_mb=round(peak_ram_mb, 2),
            final_ram_mb=round(final_ram_mb, 2),
            ram_delta_mb=round(ram_delta_mb, 2),
            storage_footprint=storage,
        )
