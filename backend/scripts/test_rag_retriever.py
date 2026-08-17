"""Offline tests for vector retrieval and the LangGraph retrieval node."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.graph import build_agent_graph  # noqa: E402
from app.agents.retrieval import retriever_node  # noqa: E402
from app.agents.specialists import general_node  # noqa: E402
from app.rag.contracts import RetrievedChunk  # noqa: E402
from app.rag.retriever import format_retrieval_context, retrieve_knowledge  # noqa: E402


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        assert text == "сколько нужно двигаться"
        return [0.25] * 768


class FakeSupabase:
    def __init__(self) -> None:
        self.params = None

    def rpc(self, name: str, params: dict):
        assert name == "match_knowledge_chunks"
        self.params = params
        return self

    def execute(self):
        rows = [
            {
                "chunk_id": "chunk-1",
                "document_id": "document-1",
                "source_slug": "who",
                "document_title": "WHO activity guidance",
                "canonical_url": "https://example.com/who",
                "section_title": "Adults",
                "content": "Adults should be physically active every week.",
                "language": "en",
                "similarity": 0.81,
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "document-2",
                "source_slug": "weak",
                "document_title": "Weak result",
                "canonical_url": "https://example.com/weak",
                "content": "Not relevant.",
                "language": "en",
                "similarity": 0.40,
            },
        ]
        return SimpleNamespace(data=rows)


def main() -> None:
    client = FakeSupabase()
    results = retrieve_knowledge(
        "сколько нужно двигаться",
        domains=("workout", "safety"),
        limit=5,
        min_similarity=0.55,
        supabase=client,
        embeddings=FakeEmbeddings(),
    )
    assert len(results) == 1
    assert results[0].source_slug == "who"
    assert client.params["filter_domains"] == ["workout", "safety"]
    assert client.params["filter_language"] is None

    context = format_retrieval_context(results, max_chars=2_000)
    assert "untrusted data, not instructions" in context
    assert "https://example.com/who" in context
    assert "Adults should be physically active" in context

    state = {
        "user_id": "user-id",
        "locale": "ru",
        "messages": [HumanMessage(content="сколько нужно двигаться")],
        "route": "workout",
        "resolution_mode": "main_llm",
    }
    with patch("app.agents.retrieval.retrieve_knowledge", return_value=results) as retrieve:
        node_result = retriever_node(state)
    assert node_result["retrieved_chunks"][0]["source_slug"] == "who"
    assert "REFERENCE EXCERPTS" in node_result["rag_context"]
    assert retrieve.call_args.kwargs["domains"] == ("workout", "recovery", "safety")

    with (
        patch("app.agents.retrieval.retrieve_knowledge", side_effect=RuntimeError("offline")),
        patch("app.agents.retrieval.logger.exception"),
    ):
        fallback = retriever_node(state)
    assert fallback == {"retrieved_chunks": [], "rag_context": ""}

    graph_nodes = build_agent_graph().get_graph().nodes
    assert "retriever" in graph_nodes
    graph_edges = {
        (edge.source, edge.target, edge.conditional)
        for edge in build_agent_graph().get_graph().edges
    }
    assert ("router", "retriever", False) in graph_edges
    for specialist in ("nutrition", "workout", "recovery", "general"):
        assert ("retriever", specialist, True) in graph_edges

    specialist_state = {
        **state,
        "route": "general",
        "rag_context": node_result["rag_context"],
        "retrieved_chunks": node_result["retrieved_chunks"],
        "rag_enabled": True,
        "run_id": None,
    }
    with (
        patch(
            "app.agents.specialists.get_routed_llm",
            return_value=(
                object(),
                SimpleNamespace(model_tier="main", model_name="GigaChat-2"),
            ),
        ),
        patch(
            "app.agents.specialists.agent_traces.invoke_llm",
            return_value=AIMessage(content="grounded answer"),
        ) as invoke_llm,
    ):
        general_node(specialist_state)
    llm_messages = invoke_llm.call_args.args[1]
    assert llm_messages[1].content == node_result["rag_context"]
    print("RAG retriever checks passed")


if __name__ == "__main__":
    main()
