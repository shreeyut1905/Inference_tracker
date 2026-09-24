from __future__ import annotations

from inference_tracker.storage import StateStore


def test_state_store_filters_and_records_sent_papers(tmp_path):
    from inference_tracker.models import Paper

    paper = Paper(canonical_id="arxiv:2609.12345", title="Paper", abstract="Abstract")
    with StateStore(str(tmp_path / "state.sqlite3")) as store:
        assert store.filter_unsent([paper]) == [paper]
        store.mark_sent([paper])
        assert store.filter_unsent([paper]) == []
        assert store.sent_ids() == {"arxiv:2609.12345"}

        rejected = Paper(canonical_id="arxiv:2609.12346", title="Rejected", abstract="Abstract")
        assert store.filter_unreviewed([rejected]) == [rejected]
        store.mark_reviewed([rejected])
        assert store.filter_unreviewed([rejected]) == []
        assert store.reviewed_ids() == {"arxiv:2609.12346"}
