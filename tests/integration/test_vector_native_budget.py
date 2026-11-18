"""Native vector row ceilings refuse complete reads and release their credits."""

from __future__ import annotations

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from server.core import ch
from server.core.ch_errors import error_code
from server.core.errors import VectorAdmissionError
from server.core.vectors import admission
from server.settings import settings
from tests.integration._resources import unique_org

pytestmark = pytest.mark.usefixtures("ch_target_required")


async def test_native_row_ceiling_refuses_the_read_and_releases_credits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "vector_scan_max_rows_to_read", 16)
    monkeypatch.setattr(settings, "vector_scan_max_concurrent_per_org", 2)
    monkeypatch.setattr(settings, "vector_scan_max_waiters_per_org", 2)
    monkeypatch.setattr(settings, "ch_pool_maxsize", 2)
    monkeypatch.setattr(settings, "ch_pool_wait_ms", 500)
    org = unique_org("vector-native-budget")
    sql = "SELECT sum(number) AS total FROM numbers({rows:UInt64})"
    client = ch.CHClient()
    gate = admission._admission
    population_before = gate.population
    pool_before = ch.pool_reading()
    work_before = admission.current_work()
    try:
        with ch.query_tag("itest.vector-native-budget"), admission.vector_work("", (), 0):
            await client.connect()
            assert await client.query_all(sql, {"rows": 64}) == [{"total": 2016}]
            with admission.vector_work(org, "native-budget", weight=2):
                result: object = None
                with pytest.raises(VectorAdmissionError) as caught:
                    result = await client.query_all(sql, {"rows": 64})
                assert result is None
                assert type(caught.value) is VectorAdmissionError
                assert caught.value.code == "VECTOR_CAPACITY_REFUSED"
                assert caught.value.status_code == 429
                assert caught.value.extra == {"reason": "rows", "retry_after_s": 1}
                assert caught.value.headers() == {"Retry-After": "1"}
                assert isinstance(caught.value.__cause__, DatabaseError)
                assert error_code(caught.value.__cause__) == 158
                assert gate.population == population_before
                assert org not in gate.orgs
                assert ch.pool_reading() == pool_before
                assert client._inflight == {}
                assert client._retired == {}

                assert await client.query_all(sql, {"rows": 4}) == [{"total": 6}]
            assert admission.current_work() is None
            assert gate.population == population_before
            assert org not in gate.orgs
            assert ch.pool_reading() == pool_before
            assert client._inflight == {}
            assert client._retired == {}
    finally:
        await client.disconnect()
    assert admission.current_work() == work_before
