"""Pydantic models for the remote-training API."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


GPU_TYPE_SELECTOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
NODE_SELECTOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-,\[\]]*")


class NetworkPayload(BaseModel):
    """A compiled network included in a training job."""

    format: Literal["nntree"] = "nntree"
    value: dict[str, Any]


class ResourceRequest(BaseModel):
    """Resources requested by a job from a compute-unit profile."""

    cpu: int = Field(default=1, ge=0)
    memory_gb: float = Field(default=1, gt=0)
    gpu: int = Field(default=0, ge=0)
    gpu_memory_gb: float | None = Field(default=None, gt=0)
    gpu_type: str | None = None
    node: str | None = None

    @field_validator("gpu_type")
    @classmethod
    def empty_gpu_type_is_none(cls, value: str | None) -> str | None:
        """Normalize and validate a GPU selector used in ``#SBATCH``."""

        if not value:
            return None
        if not GPU_TYPE_SELECTOR.fullmatch(value):
            raise ValueError("gpu_type selector contains unsupported characters")
        return value

    @field_validator("node")
    @classmethod
    def empty_node_is_none(cls, value: str | None) -> str | None:
        """Normalize and validate a Slurm node-list selector."""

        if not value:
            return None
        if not NODE_SELECTOR.fullmatch(value):
            raise ValueError("node selector contains unsupported characters")
        return value


class JobSubmission(BaseModel):
    """Complete request submitted to the training backend."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    network: NetworkPayload
    training: dict[str, Any]
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    priority: int = Field(default=0, ge=0, le=1_000_000)


class DatasetParameter(BaseModel):
    """Metadata for one dataset constructor parameter."""

    name: str
    type: str
    default: Any = None
    required: bool = False


class DatasetInfo(BaseModel):
    """Discoverable dataset class and its constructor metadata."""

    target: str
    name: str
    doc: str = ""
    parameters: list[DatasetParameter] = Field(default_factory=list)


class ComputeUnitInfo(BaseModel):
    """Public description of a configured compute-unit profile."""

    id: str
    kind: Literal["local", "slurm"]
    capacity: ResourceRequest
    enabled: bool = True


class JobStatus(BaseModel):
    """Public job metadata returned by the API."""

    id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    priority: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    executor: str | None = None
    compute_unit: str | None = None
    error: str | None = None
    heartbeat_at: str | None = None
    wandb_url: str | None = None
    artifact_dir: str


class PairingRequestInput(BaseModel):
    """Optional browser metadata supplied when requesting a connection."""

    model_config = ConfigDict(extra="forbid")

    device_name: str | None = Field(default=None, max_length=80)


class PairingGrantResponse(BaseModel):
    """One-time connection credentials returned to the browser."""

    request_id: str
    connection_id: str
    token: str
    verification_code: str
    expires_at: str


class PairingStatusResponse(BaseModel):
    """Observable state of a browser pairing request."""

    request_id: str
    connection_id: str
    status: Literal["pending", "approved", "rejected", "expired"]
    verification_code: str
    expires_at: str
    session_expires_at: str | None = None


class SessionInfo(BaseModel):
    """Public metadata for the authenticated browser connection."""

    id: str
    device_name: str | None = None
    status: str
    created_at: str
    approved_at: str | None = None
    expires_at: str | None = None
    last_seen_at: str | None = None
    revoked_at: str | None = None


class PairingApprovalInput(BaseModel):
    """Optional lifetime override supplied by a backend administrator."""

    model_config = ConfigDict(extra="forbid")

    ttl: str | None = None
