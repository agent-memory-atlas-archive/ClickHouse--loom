"""Owning vector APIs retain bounded, complete read work."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from server.core import ch as ch_module
from server.core.errors import VectorInputTooLargeError
from server.knowledge.app.search import HybridSearchQuery, hybrid_search
from server.memory.app.search import ann_entity_boosts
from server.settings import settings


class Driver:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.started = 0
        self.release = asyncio.Event()
        self.settings: list[dict[str, Any]] = []
        self.failure: Exception | None = None

    async def query(self, sql: str, **kwargs: Any) -> Any:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started += 1
        self.settings.append(kwargs["settings"])
        try:
            await self.release.wait()
            if self.failure is not None:
                raise self.failure
            return SimpleNamespace(result_rows=[], column_names=[], summary={})
        finally:
            self.active -= 1


class Embedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def client(monkeypatch: pytest.MonkeyPatch, *, pool: int = 64) -> tuple[ch_module.CHClient, Driver]:
    monkeypatch.setattr(settings, "ch_pool_maxsize", pool)
    monkeypatch.setattr(settings, "ch_pool_wait_ms", 5000)

    async def schema_ready(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(ch_module, "guard_pending_schema", schema_ready)
    ch = ch_module.CHClient()
    driver = Driver()
    monkeypatch.setattr(ch, "_client", driver)
    return ch, driver


def query(org: str) -> HybridSearchQuery:
    return HybridSearchQuery(
        org=org, namespaces=("private",), embedding_model="test", query_vec=[1.0, 0.0], top_k=1
    )


async def settled() -> None:
    # Every fake driver suspends on an Event; no wall-clock race is required.
    for _ in range(80):
        await asyncio.sleep(0)


async def join(tasks: list[asyncio.Task[Any]], driver: Driver) -> list[Any]:
    driver.release.set()
    return await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_knowledge_read_limits_one_org_without_dropping_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ch, driver = client(monkeypatch)
    tasks = [asyncio.create_task(hybrid_search(query("one-org"), ch=ch)) for _ in range(24)]
    try:
        await settled()
        assert driver.active == 16
        assert driver.peak == 16
    finally:
        results = await join(tasks, driver)
    assert results == [[]] * 24
    assert driver.started == 24
    assert driver.active == 0


@pytest.mark.asyncio
async def test_two_axis_entity_boost_costs_two_units(monkeypatch: pytest.MonkeyPatch) -> None:
    ch, driver = client(monkeypatch)
    monkeypatch.setattr(ann_entity_boosts, "get_client", lambda: ch)
    tasks = [
        asyncio.create_task(
            ann_entity_boosts._ann_entity_boosts(
                org="one-org",
                namespaces=["private"],
                agent="a",
                user_id="u",
                query_entities=[str(i)],
                embedder=Embedder(),
            )
        )
        for i in range(12)
    ]
    try:
        await settled()
        assert driver.active == 8
        assert driver.peak == 8
    finally:
        results = await join(tasks, driver)
    assert results == [{}] * 12
    assert driver.started == 12
    assert driver.active == 0


@pytest.mark.asyncio
async def test_distinct_org_population_includes_existing_pool_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ch, driver = client(monkeypatch)
    tasks = [asyncio.create_task(hybrid_search(query(f"org-{i}"), ch=ch)) for i in range(577)]
    try:
        await settled()
        assert driver.active == 64
        assert sum(not task.done() for task in tasks) == 512
    finally:
        results = await join(tasks, driver)
    refused = [result for result in results if isinstance(result, Exception)]
    assert len(refused) == 65
    assert all(getattr(result, "code", "") == "OVERLOADED" for result in refused)
    assert results.count([]) == 512
    assert driver.started == 512
    assert driver.active == 0


@pytest.mark.asyncio
async def test_vector_read_refuses_disabled_pool_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ch, driver = client(monkeypatch, pool=0)
    driver.release.set()
    with pytest.raises(Exception) as caught:
        await hybrid_search(query("one-org"), ch=ch)
    assert getattr(caught.value, "code", "") == "OVERLOADED"
    assert driver.started == 0


@pytest.mark.asyncio
async def test_vector_native_rows_throw_cap_applies_without_traffic_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ch, driver = client(monkeypatch)
    driver.release.set()
    assert await hybrid_search(query("one-org"), ch=ch) == []
    assert driver.settings[0].get("max_rows_to_read") == 1_000_000
    assert driver.settings[0].get("read_overflow_mode") == "throw"
    assert 0 < driver.settings[0]["max_execution_time"] <= 30
    assert 0 < driver.settings[0]["max_memory_usage"] <= 4_000_000_000


@pytest.mark.asyncio
async def test_entity_boost_does_not_turn_typed_refusal_into_empty_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ch, driver = client(monkeypatch)
    driver.release.set()
    driver.failure = VectorInputTooLargeError("bounded test refusal")
    monkeypatch.setattr(ann_entity_boosts, "get_client", lambda: ch)
    with pytest.raises(VectorInputTooLargeError, match="bounded test refusal"):
        await ann_entity_boosts._ann_entity_boosts(
            org="one-org",
            namespaces=["private"],
            agent="a",
            user_id="u",
            query_entities=["entity"],
            embedder=Embedder(),
        )
    assert driver.active == 0


@pytest.mark.asyncio
async def test_entity_boost_keeps_ordinary_optional_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ch, driver = client(monkeypatch)
    driver.release.set()
    driver.failure = RuntimeError("ordinary optional failure")
    monkeypatch.setattr(ann_entity_boosts, "get_client", lambda: ch)
    assert (
        await ann_entity_boosts._ann_entity_boosts(
            org="one-org",
            namespaces=["private"],
            agent="a",
            user_id="u",
            query_entities=["entity"],
            embedder=Embedder(),
        )
        == {}
    )
    assert driver.active == 0


@pytest.mark.asyncio
async def test_nonvector_read_retains_existing_pool_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ch, driver = client(monkeypatch, pool=0)
    driver.release.set()
    assert await ch.query_all("SELECT 1") == []
    assert driver.started == 1


class NativeBudgetError(RuntimeError):
    def __init__(self, code: int) -> None:
        super().__init__(f"Code: {code}. Native budget test refusal")
        self.code = code


@pytest.mark.asyncio
@pytest.mark.parametrize("code,reason", [(158, "rows"), (241, "memory")])
async def test_entity_boost_native_budget_is_a_complete_failure(
    monkeypatch: pytest.MonkeyPatch, code: int, reason: str
) -> None:
    ch, driver = client(monkeypatch)
    driver.release.set()
    driver.failure = NativeBudgetError(code)
    monkeypatch.setattr(ann_entity_boosts, "get_client", lambda: ch)
    with pytest.raises(Exception) as caught:
        await ann_entity_boosts._ann_entity_boosts(
            org="one-org",
            namespaces=["private"],
            agent="a",
            user_id="u",
            query_entities=["entity"],
            embedder=Embedder(),
        )
    assert getattr(caught.value, "code", "") == "VECTOR_CAPACITY_REFUSED"
    assert getattr(caught.value, "extra", {}).get("reason") == reason
    assert driver.active == 0


@pytest.mark.asyncio
async def test_selected_bulk_check_refusal_is_not_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    from server.memory.app import bulk_set, conflicts

    async def policy(*args: Any, **kwargs: Any) -> str:
        return "surface"

    async def refused(*args: Any, **kwargs: Any) -> Any:
        raise VectorInputTooLargeError("selected conflict read refused")

    monkeypatch.setattr(settings, "memory_bulk_peer_read", "per_fact")
    monkeypatch.setattr(bulk_set, "_namespace_policy", policy)
    monkeypatch.setattr(conflicts, "check_on_commit", refused)
    input = SimpleNamespace(org="one-org", namespace="private", agent="a", user_id="u")
    row = {"memory_type": "semantic", "memory_key": "fact"}
    entry = SimpleNamespace(memory_type="semantic", memory_key="fact")
    with pytest.raises(VectorInputTooLargeError, match="selected conflict read refused"):
        await bulk_set._bulk_conflict_verdicts(
            input,
            [row],
            [(entry, "value", None, None, None)],
            [(0, None, None, None)],
            [[1.0, 0.0]],
            floors={"semantic": (0.8, 0.8)},
        )


@pytest.mark.asyncio
async def test_postcommit_refusal_preserves_versions_and_reports_optional_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.memory.app import bulk_set, key_supersession, value_updates

    async def no_retirement(**kwargs: Any) -> None:
        return None

    async def refused(**kwargs: Any) -> Any:
        raise VectorInputTooLargeError("optional linking read refused")

    monkeypatch.setattr(key_supersession, "supersede_same_key_rows", no_retirement)
    monkeypatch.setattr(value_updates, "link_value_updates", refused)
    input = SimpleNamespace(org="one-org", namespace="private", agent="a", detect_updates=True)
    row = {
        "org": "one-org",
        "namespace": "private",
        "agent": "a",
        "user_id": "u",
        "session_id": "s",
        "memory_key": "fact",
        "version": 7,
        "status": "active",
    }
    result = SimpleNamespace(memory_key="fact", version=7, next="Existing next.")
    state = SimpleNamespace(memory_rows=[row], embedding_rows=[], results=[result])
    assert await bulk_set._bulk_post_insert_links(None, input, state, []) == []
    assert result.version == 7
    assert result.next.startswith("Existing next.")
    assert "vector capacity" in result.next
    assert "committed" in result.next


def test_vector_scan_waiter_setting_accepts_one_and_refuses_invalid_values() -> None:
    from pydantic import ValidationError

    from server.settings import Settings

    assert Settings(vector_scan_max_waiters_per_org=1).vector_scan_max_waiters_per_org == 1
    for invalid in (0, -1, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError, match="vector_scan_max_waiters_per_org"):
            Settings.model_validate({"vector_scan_max_waiters_per_org": invalid})


def test_vector_scan_row_setting_accepts_one_and_refuses_invalid_values() -> None:
    from pydantic import ValidationError

    from server.settings import Settings

    assert Settings(vector_scan_max_rows_to_read=1).vector_scan_max_rows_to_read == 1
    for invalid in (0, -1, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError, match="vector_scan_max_rows_to_read"):
            Settings.model_validate({"vector_scan_max_rows_to_read": invalid})


@pytest.mark.asyncio
async def test_vector_waiter_setting_bounds_queued_reads_and_reclaims_cancelled_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.core.errors import VectorCapacityError

    monkeypatch.setattr(settings, "vector_scan_max_concurrent_per_org", 2)
    monkeypatch.setattr(settings, "vector_scan_max_waiters_per_org", 4)
    ch, driver = client(monkeypatch)
    tasks = [asyncio.create_task(hybrid_search(query("one-org"), ch=ch)) for _ in range(6)]
    try:
        await settled()
        assert driver.active == 2
        assert sum(not task.done() for task in tasks) == 6
        with pytest.raises(VectorCapacityError) as raised:
            await hybrid_search(query("one-org"), ch=ch)
        assert raised.value.extra["reason"] == "org_queue"
        assert driver.started == 2

        tasks[-1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[-1]
        replacement = asyncio.create_task(hybrid_search(query("one-org"), ch=ch))
        tasks.append(replacement)
        await settled()
        assert not replacement.done()
        assert driver.started == 2
    finally:
        results = await join(tasks, driver)
    assert results.count([]) == 6
    assert sum(isinstance(result, asyncio.CancelledError) for result in results) == 1
    assert driver.started == 6
    assert driver.active == 0

    # Both org credits are usable after refusal, queued cancellation and drain.
    driver.release.clear()
    probes = [asyncio.create_task(hybrid_search(query("one-org"), ch=ch)) for _ in range(2)]
    try:
        await settled()
        assert driver.active == 2
    finally:
        probe_results = await join(probes, driver)
    assert probe_results == [[], []]
    assert driver.started == 8
    assert driver.active == 0


@pytest.mark.asyncio
async def test_vector_row_setting_reaches_the_actual_native_read_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "vector_scan_max_rows_to_read", 1234)
    ch, driver = client(monkeypatch)
    driver.release.set()

    assert await hybrid_search(query("one-org"), ch=ch) == []
    assert driver.started == 1
    assert driver.settings[0]["max_rows_to_read"] == 1234
    assert driver.settings[0]["read_overflow_mode"] == "throw"
    assert driver.active == 0


@pytest.mark.parametrize("kind", ["admission", "input", "overload"])
def test_vector_concrete_error_hierarchy_preserves_wire_contracts(kind: str) -> None:
    from server.core.errors import (
        LoomServerError,
        OverloadedError,
        ValidationError,
        VectorAdmissionError,
        VectorCapacityError,
        VectorOverloadedError,
        classified,
    )

    if kind == "admission":
        error = VectorAdmissionError("refused", reason="org_queue", retry_after_s=3)
        public_family = LoomServerError
        expected = (
            429,
            {
                "code": "VECTOR_CAPACITY_REFUSED",
                "message": "refused",
                "reason": "org_queue",
                "retry_after_s": 3,
            },
            {"Retry-After": "3"},
        )
        category, retryable = "capacity", True
    elif kind == "input":
        error = VectorInputTooLargeError("too many seeds")
        public_family = ValidationError
        expected = (422, {"code": "VECTOR_INPUT_TOO_LARGE", "message": "too many seeds"}, None)
        category, retryable = "input", False
    else:
        error = VectorOverloadedError("busy", reason="population", retry_after_s=2)
        public_family = OverloadedError
        expected = (
            503,
            {"code": "OVERLOADED", "message": "busy", "reason": "population", "retry_after_s": 2},
            {"Retry-After": "2"},
        )
        category, retryable = "capacity", True

    assert isinstance(error, public_family)
    with pytest.raises(VectorCapacityError) as caught_capacity:
        raise error
    assert caught_capacity.value is error
    with pytest.raises(LoomServerError) as caught_wire:
        raise error
    assert caught_wire.value is error
    assert (error.status_code, error.to_detail(), error.headers()) == expected
    assert classified(error.to_detail(), error.status_code) == {
        **expected[1],
        "category": category,
        "retryable": retryable,
    }
